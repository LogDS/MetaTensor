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
#include <logds/metatensor/StorageLayout.h>
#include <logds/metatensor/DistributedContext.h>

// Forward Declaration del Nastro fortemente tipizzato
template <typename... TensorTypes> struct GradientTape;

enum class OptimizerType { SGD, Momentum, Adam };
enum class DecayType { None, Step, Exponential };

struct OptimizerState {
    OptimizerType type = OptimizerType::SGD;
    std::vector<torch::Tensor> exp_avg;
    std::vector<torch::Tensor> exp_avg_sq;
    int64_t step = 0;
    float momentum_beta = 0.9f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float eps = 1e-8f;
    DecayType decay_scheme = DecayType::None;
    float lr_gamma = 0.95f;
    int64_t decay_steps = 10;
};

inline auto DefaultEarlyStop = [](float loss) -> bool {
    return std::isnan(loss) || std::isinf(loss);
};

// =============================================================================
// CONTESTO DELL'EPOCA: Eredita i medesimi tipi del Nastro Padre
// =============================================================================
template <typename EarlyStopLambda, typename... TensorTypes>
struct EpochContext {
    std::tuple<TensorTypes*...> watched_tensors;
    float learning_rate;
    torch::Tensor loss_snapshot; // RISOLUTIVO: Copia shallow protetta invece del puntatore orfano
    bool* early_stop_flag = nullptr;
    EarlyStopLambda early_stop_predicate;

    GradientTape<TensorTypes...>* tape_parent_ptr = nullptr;

    template <typename LossTensorT>
    void feed_loss(const LossTensorT& loss_tensor) {
        // RISOLUTIVO: Eseguiamo una copia shallow del tensore di LibTorch.
        // Questo incrementa il contatore dei riferimenti interno ad ATen,
        // impedendo allo Zero-Caching di distruggere il grafo prima del backward.
        if (loss_tensor.storage.defined()) {
            loss_snapshot = loss_tensor.storage;
        }

        float current_loss = static_cast<float>(loss_tensor);
        if (early_stop_predicate(current_loss)) {
            if (early_stop_flag) *early_stop_flag = true;
        }
    }

    explicit operator bool() const {
        if (early_stop_flag && *early_stop_flag) return false;
        return true;
    }

    ~EpochContext();
};

// =============================================================================
// GRADIENT TAPE: Il blocco variadic viene congelato qui all'atto della creazione
// =============================================================================
template <typename... TensorTypes>
struct GradientTape {
    std::tuple<TensorTypes*...> watched_tensors; // Memorizzazione centralizzata dei puntatori
    std::vector<torch::Tensor*> watched_storages;
    OptimizerState optimizer_state;

    GradientTape(TensorTypes&... tensors) : watched_tensors(std::make_tuple(&tensors...)) {
        torch::autograd::GradMode::set_enabled(true);
        ([&]() {
            tensors.watch();
            watched_storages.push_back(&tensors.storage);
        }(), ...);
    }

    void set_optimizer(OptimizerType type) { optimizer_state.type = type; }
    void set_lr_decay(DecayType scheme, float gamma, int64_t steps = 10) {
        optimizer_state.decay_scheme = scheme;
        optimizer_state.lr_gamma = gamma;
        optimizer_state.decay_steps = steps;
    }

    // Aggiornamento centralizzato: non è più una funzione template indipendente!
        // All'interno di struct GradientTape in GradientTape.h
    // All'interno di struct GradientTape in GradientTape.h
        // =============================================================================
    // MOTORE DI OTTIMIZZAZIONE AVANZATO CENTRALIZZATO (C++26 Variadic Framework)
    // =============================================================================
    // Esegue il calcolo e la sincronizzazione multinodo dei gradienti, applica
    // le equazioni dei momenti hardware e pulisce deterministicamente la VRAM.
    void apply_optimization_step(float base_lr, std::tuple<TensorTypes*...> watched_tensors) {
        optimizer_state.step++;

        // 1. CALCOLO DINAMICO DEL LEARNING RATE DECAY
        float current_lr = base_lr;
        if (optimizer_state.decay_scheme == DecayType::Exponential) {
            current_lr = base_lr * std::pow(optimizer_state.lr_gamma, static_cast<float>(optimizer_state.step));
        }
        else if (optimizer_state.decay_scheme == DecayType::Step) {
            int64_t intervals = optimizer_state.step / optimizer_state.decay_steps;
            current_lr = base_lr * std::pow(optimizer_state.lr_gamma, static_cast<float>(intervals));
        }

        // 2. INIZIALIZZAZIONE LAZY DEI BUFFER DEI MOMENTI STORICI IN VRAM
        size_t num_tensors = sizeof...(TensorTypes);
        if (optimizer_state.exp_avg.empty()) {
            optimizer_state.exp_avg.resize(num_tensors);
            if (optimizer_state.type == OptimizerType::Adam) {
                optimizer_state.exp_avg_sq.resize(num_tensors);
            }

            size_t idx = 0;
            std::apply([&](auto*... tensor_ptrs) {
                (([&]() {
                    if (tensor_ptrs) {
                        using CurrentTensorT = std::remove_pointer_t<decltype(tensor_ptrs)>;

                        // Se il tensore è sparso e usiamo Adam, forziamo i momenti a essere allocati come densi
                        if (CurrentTensorT::layout == StorageLayout::SparseCOO && optimizer_state.type == OptimizerType::Adam) {
                            optimizer_state.exp_avg[idx] = torch::zeros(tensor_ptrs->storage.sizes(), tensor_ptrs->storage.options().layout(torch::kStrided));
                            optimizer_state.exp_avg_sq[idx] = torch::zeros(tensor_ptrs->storage.sizes(), tensor_ptrs->storage.options().layout(torch::kStrided));
                        } else {
                            optimizer_state.exp_avg[idx] = torch::zeros_like(tensor_ptrs->storage);
                            if (optimizer_state.type == OptimizerType::Adam) {
                                optimizer_state.exp_avg_sq[idx] = torch::zeros_like(tensor_ptrs->storage);
                            }
                        }
                    }
                    idx++;
                }()), ...);
            }, watched_tensors);
        }

        // 3. SINCRONIZZAZIONE ED AGGIORNAMENTO HARDWARE PARAMETRICO
        size_t tensor_idx = 0;
        std::apply([&](auto*... tensor_ptrs) {
            (([&]() {
                if (tensor_ptrs) {
                    torch::NoGradGuard no_grad;

                    // A) Estrazione del gradiente grezzo di LibTorch generato dall'Autograd
// =============================================================================
                    // ESTRAZIONE E ISOLAMENTO DEL GRADIENTE (Risoluzione Loss = 13 in Locale)
                    // =============================================================================
                    // =============================================================================
                    // ESTRAZIONE E ACCASAMENTO ISOLATO DEL GRADIENTE (Risoluzione Errori di Build)
                    // =============================================================================
                    torch::Tensor raw_grad = tensor_ptrs->storage.grad();
                    if (!raw_grad.defined()) {
                        raw_grad = torch::zeros_like(tensor_ptrs->storage);
                    }

                    using CurrentTensorT = std::remove_pointer_t<decltype(tensor_ptrs)>;

                    // RISOLUTIVO: Istanziamo il wrapper ereditando l'esatto metatipo completo originale.
                    // Passiamo il gradiente staccato (.detach()) per blindare la memoria in locale.
                    CurrentTensorT local_grad_wrapper(raw_grad.detach());

                    // Eseguiamo l'AllReduce funzionale universale (restituisce il clone o lancia la rete MPI)
                    auto synced_grad_tensor = local_grad_wrapper.distributed_allreduce_sum();
                    torch::Tensor grad = synced_grad_tensor.storage;

                    // Calcoliamo la media del gradiente solo se siamo all'interno di un cluster reale
                    if (DistributedContext::is_distributed()) {
                        float w_size = static_cast<float>(DistributedContext::get_world_size());
                        grad = grad / w_size;
                    }

                    {
                        torch::NoGradGuard no_grad_set;
                        // RISOLUTIVO: Utilizziamo mutable_grad() per bypassare il vincolo const del compilatore
                        tensor_ptrs->storage.mutable_grad() = grad;
                    }

                    // =============================================================================
                    // DA QUI IN POI PROSEGUE CON LE EQUAZIONI DI ADAM/MOMENTUM/SGD (Usa 'grad')
                    // =============================================================================
                    if constexpr (CurrentTensorT::layout == StorageLayout::SparseCOO) {
                        if (optimizer_state.type == OptimizerType::SGD) {
                            tensor_ptrs->storage.sub_(grad * current_lr);
                        }
                        else {
                            // Protezione hardware per Adam/Momentum mista su matrici sparse
                            auto dense_tensor = tensor_ptrs->storage.to_dense();
                            auto dense_grad = grad.to_dense();

                            if (optimizer_state.type == OptimizerType::Momentum) {
                                optimizer_state.exp_avg[tensor_idx].mul_(optimizer_state.momentum_beta).add_(dense_grad);
                                dense_tensor.sub_(optimizer_state.exp_avg[tensor_idx] * current_lr);
                            }
                            else if (optimizer_state.type == OptimizerType::Adam) {
                                optimizer_state.exp_avg[tensor_idx].mul_(optimizer_state.adam_beta1).add_(dense_grad * (1.0f - optimizer_state.adam_beta1));
                                optimizer_state.exp_avg_sq[tensor_idx].mul_(optimizer_state.adam_beta2).add_(dense_grad.pow(2) * (1.0f - optimizer_state.adam_beta2));

                                float bc1 = 1.0f - std::pow(optimizer_state.adam_beta1, optimizer_state.step);
                                float bc2 = 1.0f - std::pow(optimizer_state.adam_beta2, optimizer_state.step);
                                auto step_size = current_lr / bc1;
                                auto denom = (optimizer_state.exp_avg_sq[tensor_idx].sqrt() / std::sqrt(bc2)).add_(optimizer_state.eps);

                                dense_tensor.sub_((optimizer_state.exp_avg[tensor_idx] / denom) * step_size);
                            }
                            tensor_ptrs->storage = dense_tensor.to_sparse().coalesce();
                        }
                    }
                    else {
                        // Caso standard per i tensori densi regolari
                        if (optimizer_state.type == OptimizerType::SGD) {
                            tensor_ptrs->storage.sub_(grad * current_lr);
                        }
                        else if (optimizer_state.type == OptimizerType::Momentum) {
                            optimizer_state.exp_avg[tensor_idx].mul_(optimizer_state.momentum_beta).add_(grad);
                            tensor_ptrs->storage.sub_(optimizer_state.exp_avg[tensor_idx] * current_lr);
                        }
                        else if (optimizer_state.type == OptimizerType::Adam) {
                            optimizer_state.exp_avg[tensor_idx].mul_(optimizer_state.adam_beta1).add_(grad * (1.0f - optimizer_state.adam_beta1));
                            optimizer_state.exp_avg_sq[tensor_idx].mul_(optimizer_state.adam_beta2).add_(grad.pow(2) * (1.0f - optimizer_state.adam_beta2));
                            float bc1 = 1.0f - std::pow(optimizer_state.adam_beta1, optimizer_state.step);
                            float bc2 = 1.0f - std::pow(optimizer_state.adam_beta2, optimizer_state.step);
                            auto step_size = current_lr / bc1;
                            auto denom = (optimizer_state.exp_avg_sq[tensor_idx].sqrt() / std::sqrt(bc2)).add_(optimizer_state.eps);
                            tensor_ptrs->storage.sub_((optimizer_state.exp_avg[tensor_idx] / denom) * step_size);
                        }
                    }


                    // E) ZERO-CACHING SINCRO DEI GRADIENTI RESIDUI
                    if (tensor_ptrs->storage.grad().defined()) {
                        tensor_ptrs->storage.grad().zero_();
                    }
                }
                tensor_idx++;
            }()), ...);
        }, watched_tensors);
    }



    ~GradientTape() {
        for (auto* storage_ptr : watched_storages) {
            if (storage_ptr && storage_ptr->defined()) { storage_ptr->set_requires_grad(false); }
        }
    }

    // Le funzioni passano i puntatori pre-congelati nel costruttore, eliminando l'ambiguità variadic
    // VARIANTE A (Nome Esplicito): Per criteri di stop arbitrari via Lambda custom
    template <typename EarlyStopLambda>
    auto next_epoch_custom(float lr, bool& early_stop, EarlyStopLambda&& custom_predicate) {
        // RISOLUTIVO: Rimosso 'nullptr' per loss_snapshot.
        // Venendo omesso, torch::Tensor si inizializza automaticamente come istanza vuota/indefinita.
        return EpochContext<std::decay_t<EarlyStopLambda>, TensorTypes...>{
            watched_tensors, lr, torch::Tensor(), &early_stop, std::forward<EarlyStopLambda>(custom_predicate), this
        };
    }

    // VARIANTE B (Nome Standard): Fallback automatico con DefaultEarlyStop (NaN/Inf)
    auto next_epoch(float lr, bool& early_stop) {
        // RISOLUTIVO: Rimosso 'nullptr' per loss_snapshot.
        return EpochContext<decltype(DefaultEarlyStop), TensorTypes...>{
            watched_tensors, lr, torch::Tensor(), &early_stop, DefaultEarlyStop, this
        };
    }
};

// Implementazione differita del distruttore ancorata alla classe unificata
// In fondo a GradientTape.h, subito prima di #endif
template <typename EarlyStopLambda, typename... TensorTypes>
EpochContext<EarlyStopLambda, TensorTypes...>::~EpochContext() {
    if ((early_stop_flag && *early_stop_flag) || !loss_snapshot.defined()) {
        return;
    }

    // 1. Lancio del backward pass sul grafo hardware
    loss_snapshot.backward();

    // 2. RISOLUTIVO: Allineamento dei parametri passati alla chiamata centralizzata del nastro.
    // Inseriamo 'watched_tensors' come richiesto dalla firma del metodo apply_optimization_step.
    if (tape_parent_ptr) {
        tape_parent_ptr->apply_optimization_step(learning_rate, watched_tensors);
    }
}



#endif //TENSORLIBRARY_GRADIENTTAPE_H
