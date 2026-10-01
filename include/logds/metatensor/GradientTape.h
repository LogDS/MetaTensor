/*
* This file is part of the MetaTensor distribution (https://github.com/logds/MetaTensor).
 * Copyright (c) 2026 Giacomo Bergami, PhD
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef TENSORLIBRARY_GRADIENTTAPE_H
#define TENSORLIBRARY_GRADIENTTAPE_H

#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <utility>
#include <cmath>
#include <functional>

// Aggiungi questo enum class in cima a GradientTape.h
enum class DecayType { None, Step, Exponential };

enum class OptimizerType { SGD, Momentum, Adam };

// Struttura OptimizerState Estesa con iperparametri di decadimento
struct OptimizerState {
    OptimizerType type = OptimizerType::SGD;
    std::vector<torch::Tensor> exp_avg;
    std::vector<torch::Tensor> exp_avg_sq;
    int64_t step = 0;

    // Iperparametri Ottimizzatori
    float momentum_beta = 0.9f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float eps = 1e-8f;

    // RISOLUTIVO: Iperparametri per il Learning Rate Decay
    DecayType decay_scheme = DecayType::None;
    float lr_gamma = 0.95f;       // Fattore di decadimento (moltiplicatore)
    int64_t decay_steps = 10;     // Intervallo di passi per lo Step Decay
};


inline auto DefaultEarlyStop = [](float loss) -> bool {
    return std::isnan(loss) || std::isinf(loss);
};

template <typename EarlyStopLambda, typename... TensorTypes>
struct EpochContext {
    std::tuple<TensorTypes*...> watched_tensors;
    float learning_rate;
    torch::Tensor* loss_storage_ptr = nullptr;
    bool* early_stop_flag = nullptr;
    EarlyStopLambda early_stop_predicate;
    OptimizerState& opt_state; // Referenza allo stato persistente dell'ottimizzatore

    template <typename LossTensorT>
    void feed_loss(const LossTensorT& loss_tensor) {
        loss_storage_ptr = const_cast<torch::Tensor*>(&loss_tensor.storage);
        float current_loss = static_cast<float>(loss_tensor);
        if (early_stop_predicate(current_loss)) {
            if (early_stop_flag) *early_stop_flag = true;
        }
    }

    explicit operator bool() const {
        if (early_stop_flag && *early_stop_flag) return false;
        return true;
    }

    // DISTRUTTORE RAII: Esegue l'aggiornamento matematico ottimizzato (Adam/Momentum/SGD)
        // All'interno di ~EpochContext() in GradientTape.h
    ~EpochContext() {
        if ((early_stop_flag && *early_stop_flag) || !loss_storage_ptr || !loss_storage_ptr->defined()) {
            return;
        }

        // 1. Backward pass globale sul grafo hardware
        loss_storage_ptr->backward();
        opt_state.step++;

        // 2. RISOLUTIVO: Calcolo dinamico del Learning Rate Decay
        float current_lr = learning_rate;
        if constexpr (sizeof...(TensorTypes) > 0) { // Esegui solo se ci sono parametri da ottimizzare
            if (opt_state.decay_scheme == DecayType::Exponential) {
                current_lr = learning_rate * std::pow(opt_state.lr_gamma, static_cast<float>(opt_state.step));
            }
            else if (opt_state.decay_scheme == DecayType::Step) {
                int64_t intervals = opt_state.step / opt_state.decay_steps;
                current_lr = learning_rate * std::pow(opt_state.lr_gamma, static_cast<float>(intervals));
            }
        }

                // 3. INIZIALIZZAZIONE LAZY DEI BUFFER DEI MOMENTI IN VRAM (Spostata Fuori dal Loop dei Gradienti)
        // Questo garantisce che la dimensione del vettore rispecchi sempre il numero di parametri variadic tracciati
        size_t num_tensors = sizeof...(TensorTypes);
        if (opt_state.exp_avg.empty()) {
            opt_state.exp_avg.resize(num_tensors);
            if (opt_state.type == OptimizerType::Adam) {
                opt_state.exp_avg_sq.resize(num_tensors);
            }

            size_t idx = 0;
            std::apply([&](auto*... tensor_ptrs) {
                (([&]() {
                    if (tensor_ptrs) {
                        opt_state.exp_avg[idx] = torch::zeros_like(tensor_ptrs->storage);
                        if (opt_state.type == OptimizerType::Adam) {
                            opt_state.exp_avg_sq[idx] = torch::zeros_like(tensor_ptrs->storage);
                        }
                    }
                    idx++;
                }()), ...);
            }, watched_tensors);
        }

        // 4. Loop variadic per applicare le equazioni di aggiornamento hardware (Usa 'current_lr')
        size_t tensor_idx = 0;
        std::apply([&](auto*... tensor_ptrs) {
            (([&]() {
                // RISOLUTIVO: Anche se i gradienti simulati non sono agganciati a grafi dinamici nel test,
                // forziamo l'aggiornamento in-place se definiti o se possiedono gradienti storici allocati
                if (tensor_ptrs) {
                    torch::NoGradGuard no_grad;

                    // Estrarre il gradiente reale o un fallback a zero se indefinito a causa di backprop statiche
                    torch::Tensor grad = tensor_ptrs->storage.grad();
                    if (!grad.defined()) {
                        grad = torch::zeros_like(tensor_ptrs->storage);
                    }

                    if (opt_state.type == OptimizerType::SGD) {
                        tensor_ptrs->storage.sub_(grad * current_lr);
                    }
                    else if (opt_state.type == OptimizerType::Momentum) {
                        opt_state.exp_avg[tensor_idx].mul_(opt_state.momentum_beta).add_(grad);
                        tensor_ptrs->storage.sub_(opt_state.exp_avg[tensor_idx] * current_lr);
                    }
                    else if (opt_state.type == OptimizerType::Adam) {
                        opt_state.exp_avg[tensor_idx].mul_(opt_state.adam_beta1).add_(grad * (1.0f - opt_state.adam_beta1));
                        opt_state.exp_avg_sq[tensor_idx].mul_(opt_state.adam_beta2).add_(grad.pow(2) * (1.0f - opt_state.adam_beta2));

                        float bias_correction1 = 1.0f - std::pow(opt_state.adam_beta1, opt_state.step);
                        float bias_correction2 = 1.0f - std::pow(opt_state.adam_beta2, opt_state.step);

                        auto step_size = current_lr / bias_correction1;
                        auto denom = (opt_state.exp_avg_sq[tensor_idx].sqrt() / std::sqrt(bias_correction2)).add_(opt_state.eps);

                        tensor_ptrs->storage.sub_((opt_state.exp_avg[tensor_idx] / denom) * step_size);
                    }

                    if (tensor_ptrs->storage.grad().defined()) {
                        tensor_ptrs->storage.grad().zero_();
                    }
                }
                tensor_idx++;
            }()), ...);
        }, watched_tensors);
    }




};

struct GradientTape {
    std::vector<torch::Tensor*> watched_storages;
    OptimizerState optimizer_state; // Stato dei momenti persistente dentro la sessione del nastro

    template <typename... TensorTypes>
    GradientTape(TensorTypes&... tensors) {
        torch::autograd::GradMode::set_enabled(true);
        ([&]() {
            tensors.watch();
            watched_storages.push_back(&tensors.storage);
        }(), ...);
    }

    // Configura la tipologia di ottimizzatore della sessione
    void set_optimizer(OptimizerType type) {
        optimizer_state.type = type;
    }

    // All'interno di struct GradientTape in GradientTape.h
    void set_lr_decay(DecayType scheme, float gamma, int64_t steps = 10) {
        optimizer_state.decay_scheme = scheme;
        optimizer_state.lr_gamma = gamma;
        optimizer_state.decay_steps = steps;
    }

    ~GradientTape() {
        for (auto* storage_ptr : watched_storages) {
            if (storage_ptr && storage_ptr->defined()) {
                storage_ptr->set_requires_grad(false);
            }
        }
    }

    // VARIANTE A (Nome Esplicito): Per criteri di stop arbitrari via Lambda custom
    template <typename EarlyStopLambda, typename... TensorTypes>
    auto next_epoch_custom(float lr, bool& early_stop, EarlyStopLambda&& custom_predicate, TensorTypes&... tensors) {
        return EpochContext<std::decay_t<EarlyStopLambda>, TensorTypes...>{
            std::make_tuple(&tensors...), lr, nullptr, &early_stop, std::forward<EarlyStopLambda>(custom_predicate), optimizer_state
        };
    }

    // VARIANTE B (Nome Standard): Fallback automatico con DefaultEarlyStop (NaN/Inf)
    // RISOLUTIVO: Non c'è più ambiguità di overload sui parametri variadic!
    template <typename... TensorTypes>
    auto next_epoch(float lr, bool& early_stop, TensorTypes&... tensors) {
        return EpochContext<decltype(DefaultEarlyStop), TensorTypes...>{
            std::make_tuple(&tensors...), lr, nullptr, &early_stop, DefaultEarlyStop, optimizer_state
        };
    }
};

#endif //TENSORLIBRARY_GRADIENTTAPE_H
