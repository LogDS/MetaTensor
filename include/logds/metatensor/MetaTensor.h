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

#ifndef TENSORLIBRARY_METATENSOR_H
#define TENSORLIBRARY_METATENSOR_H

#include <logds/metatensor/TensorCellProxy.h>
#include <logds/metatensor/CellOp.h>
#include <logds/metatensor/Init.h>
#include <torch/torch.h>
#include <array>
#include <iostream>
#include <utility>
#include <vector>
#include <tuple>
#include <c10/cuda/CUDACachingAllocator.h>

#include <logds/metatensor/BroadcastShape.h>
#include <logds/metatensor/RelationalEinsumCompiler.h>

#include <logds/metatensor/StorageLayout.h>

// RISOLUTIVO: Incluso qui sotto in modo che conosca i tipi e le enumerazioni definiti sopra
#include <logds/metatensor/GradientTape.h>

// =============================================================================
// IL WRAPPER DEL TENSORE: `MetaTensor` (Stile Eigen)
// =============================================================================
template <typename T, StorageLayout Layout, size_t... Dims>
class MetaTensor {
public:
    static constexpr size_t Rank = sizeof...(Dims);
    static constexpr std::array<size_t, Rank> Shape = { Dims... };
    static constexpr StorageLayout layout = Layout;
    // Inserisci in cima alla classe MetaTensor in MetaTensor.h
    static constexpr StorageLayout layout_type_dense = StorageLayout::Dense;
    static constexpr StorageLayout layout_type_sparse = StorageLayout::SparseCOO;

    // RISOLUTIVO: Espone il tipo primitivo delle celle per i riflessi del GradientTape
    using value_type = T;

    // Il tensore di LibTorch interno ereditato
    torch::Tensor storage;

    // Costruttore: alloca la memoria fisica del tensore direttamente sul device target
    // =============================================================================
    // COSTRUTTORI AVANZATI CON INIZIALIZZAZIONE STRUTTURATA (InitPattern)
    // =============================================================================

    // Costruttore flessibile guidato dai Tag Hardware
    // Costruttore flessibile guidato dai Tag Hardware con Auto-Layout Dispatching
    MetaTensor(torch::Device device = torch::kCPU,
               InitPattern pattern = InitPattern::RandomNormal) {
        std::vector<int64_t> torch_shape;
        for (size_t d : Shape) torch_shape.push_back(static_cast<int64_t>(d));
        auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

        switch (pattern) {
            case InitPattern::RandomNormal:  storage = torch::randn(torch_shape, options); break;
            case InitPattern::RandomUniform: storage = torch::rand(torch_shape, options); break;
            case InitPattern::Zeros:         storage = torch::zeros(torch_shape, options); break;
            case InitPattern::OnOnes:        storage = torch::ones(torch_shape, options); break;
            case InitPattern::Identity:
                assert(Rank == 2);
                assert(Shape[0] == Shape[1]);
                storage = torch::eye(static_cast<int64_t>(Shape[0]), options);
                break;
        }

        // RISOLUTIVO: Forza la sparsificazione immediata se richiesto dal metatipo
        if constexpr (Layout == StorageLayout::SparseCOO) {
            storage = storage.to_sparse().coalesce();
        }
    }


    // =============================================================================
    // REVISIONE COSTRUTTORE SPARSO COORDINATO (C++26 Tuple Monomorphization)
    // =============================================================================
    template <typename TupleT>
    MetaTensor(const std::vector<TupleT>& entries,
               const std::vector<T>& values,
               torch::Device device = torch::kCPU) {
        // A) Vincolo Formale a tempo di compilazione: il layout del template deve essere SparseCOO
        static_assert(Layout == StorageLayout::SparseCOO,
            "[ERR_LAYOUT] Questo costruttore è riservato esclusivamente ai tensori con layout StorageLayout::SparseCOO.");

        // B) Vincolo di Rango: la tupla inserita deve mappare esattamente il numero di assi statici
        static_assert(std::tuple_size_v<TupleT> == Rank,
            "[ERR_RANK_MISMATCH] Le tuple degli indici devono avere lo stesso numero di dimensioni del Rank del MetaTensor.");

        // C) Verifica di Coerenza a runtime delle dimensioni dei vettori
        if (entries.size() != values.size()) {
            throw std::invalid_argument("[ERR_SIZE] Il vettore delle coordinate e il vettore dei valori devono avere la stessa dimensione.");
        }

        size_t num_elements = entries.size();
        std::vector<int64_t> torch_shape(Shape.begin(), Shape.end());

        // Se non ci sono elementi, allochiamo un tensore sparso vuoto della forma geometrica corretta
        if (num_elements == 0) {
            auto empty_indices = torch::empty({static_cast<int64_t>(Rank), 0}, torch::kInt64);
            auto empty_values = torch::empty({0}, torch::kFloat32);
            storage = torch::sparse_coo_tensor(empty_indices, empty_values, torch_shape, torch::TensorOptions().device(device));
            return;
        }

        // D) SROTOLAMENTO DELLE TUPLE IN UN BUFFER LINEARE PER IL BACKEND DI LIBTORCH
        // LibTorch si aspetta gli indici in un layout bidimensionale [Rank, NumElementi]
        std::vector<int64_t> flattened_indices(Rank * num_elements);

        for (size_t col = 0; col < num_elements; ++col) {
            // Sfruttiamo una lambda helper con fold expression per srotolare la tupla in modo constexpr
            std::apply([&](const auto&... tuple_elements) {
                size_t row = 0;
                // Copia in-place degli indici convertendoli nell'offset lineare corretto della matrice dei blocchi
                ((flattened_indices[(row++) * num_elements + col] = static_cast<int64_t>(tuple_elements)), ...);
            }, entries[col]);
        }

        // E) TRASFERIMENTO BARE-METAL DEI BUFFER FISICI SUL DEVICE HARDWARE
        // Creiamo i tensori di supporto densi per indici e valori effettuando il push in VRAM
        auto options_idx = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto options_val = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        auto indices_tensor = torch::from_blob(flattened_indices.data(), {static_cast<int64_t>(Rank), static_cast<int64_t>(num_elements)}, options_idx).to(device);
        auto values_tensor = torch::from_blob(const_cast<T*>(values.data()), {static_cast<int64_t>(num_elements)}, options_val).to(device);

        // Generazione della IR del tensore sparso e coalescenza degli indici per l'ottimizzazione OpenXLA
        storage = torch::sparse_coo_tensor(indices_tensor, values_tensor, torch_shape, torch::TensorOptions().device(device)).coalesce();
    }

    public:
    // =============================================================================
    // FACTORY METHOD: GENERAZIONE DA SCALARE (C++26 Compile-Time Shape Injection)
    // =============================================================================
    // Prende un singolo valore scalare e restituisce un MetaTensor riempito con quel valore,
    // proiettandolo automaticamente sulle dimensioni spaziali attese (Dims...).
    // Esempio d'uso: auto T = MetaTensor<float, 16, 32>::from_scalar(5.5f, device);
    static auto from_scalar(T value, torch::Device device = torch::kCPU) {
        std::vector<int64_t> torch_shape;
        // Espansione del pacchetto variadic per raccogliere le dimensioni statiche
        if constexpr (Rank > 0) {
            for (size_t d : Shape) torch_shape.push_back(static_cast<int64_t>(d));
        } else {
            // Se Rank == 0, stiamo creando un vero scalare 0-D nativo di LibTorch
            torch_shape = {};
        }

        // Configurazione delle opzioni di allocazione hardware
        auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

        // Generiamo il tensore denso saturando le pagine di memoria con il valore passato
        // HLO equivalente: "broadcast" dello scalare sulla griglia geometrica
        torch::Tensor scalar_storage = torch::full(torch_shape, static_cast<float>(value), options);

        // Se l'utente ha richiesto un layout SparseCOO, la saturazione con uno scalare
        // diverso da zero distruggerebbe la parsimonia (sparsity). Forziamo quindi il tipo
        // di ritorno a Dense per coerenza matematica, oppure a Sparse se il valore è 0.
        if constexpr (Layout == StorageLayout::SparseCOO) {
            if (value != 0.0f) {
                // Ritorna la variante densa per preservare le prestazioni hardware
                return MetaTensor<T, StorageLayout::Dense, Dims...>(scalar_storage);
            } else {
                // Se lo scalare è 0, possiamo instanziare un vero tensore sparso COO vuoto
                return MetaTensor<T, StorageLayout::SparseCOO, Dims...>(scalar_storage.to_sparse());
            }
        } else {
            // Caso standard: restituisce il MetaTensor denso fortemente tipizzato
            return MetaTensor<T, StorageLayout::Dense, Dims...>(scalar_storage);
        }
    }


    // Costruttore di accoppiamento da un tensore LibTorch esistente
    MetaTensor(torch::Tensor t) : storage(t) {}

    // =============================================================================
    // METODO POLIMORFO TO_DENSE (C++26 Zero-Overhead Branching)
    // =============================================================================
    auto to_dense() const {
        if constexpr (Layout == StorageLayout::Dense) {
            // Se è già denso, restituisce una copia shallow veloce (Zero overhead)
            return *this;
        } else {
            // Se è sparso, esegue la densizzazione fisica sul chip hardware (CPU o GPU)
            return MetaTensor<T, StorageLayout::Dense, Dims...>(this->storage.to_dense());
        }
    }


    void watch() {
        this->storage.set_requires_grad(true);
    }

    // Rimuove il tensore dal grafo di registrazione dei gradienti
    void unwatch() {
        this->storage.set_requires_grad(false);
    }

    // Estrae in modo sicuro il gradiente corrente come nuovo MetaTensor
    auto grad() const {
        auto g_storage = this->storage.grad();
        if (!g_storage.defined()) {
            return MetaTensor<T, Layout, Dims...>(torch::zeros_like(this->storage));
        }
        return MetaTensor<T, Layout, Dims...>(g_storage);
    }

    // Aggiornamento SGD automatico
    void apply_gradient_descent(const MetaTensor<T, Layout, Dims...>& gradient_tensor, float learning_rate) {
        torch::NoGradGuard no_grad;
        this->storage.sub_(gradient_tensor.storage * learning_rate);
        if (this->storage.grad().defined()) {
            this->storage.grad().zero_();
        }
    }


    // =============================================================================
    // 1. METODO ESPLICITO DI AZZERAMENTO E UNMAPPING DELLA MEMORIA HARDWARE
    // =============================================================================
    void clear() {
        // Controlliamo se il tensore interno è effettivamente definito prima di procedere
        if (this->storage.defined()) {
            auto device_type = this->storage.device().type();

            // Sradichiamo il riferimento al tensore riassegnandolo a un'istanza vuota (null)
            this->storage = torch::Tensor();

            // Se il tensore risiedeva sulla GPU, forziamo il bypass del caching pool
            if (device_type == torch::kCUDA) {
                // Invia l'unmapping sincrono immediato al driver grafico (cudaFree hardware)
                // RISOLUTIVO: Chiamata formale all'allocatore statico core di LibTorch [1]
                c10::cuda::CUDACachingAllocator::emptyCache();

                // Nota alternativa: è equivalente all'uso di: at::cuda::emptyCache(); [1]
            }
        }
    }

    // =============================================================================
    // 2. DISTRUTTORE RAII: Garanzia Anti-Caching Automatica dello Scope
    // =============================================================================
    ~MetaTensor() {
        // Delegiamo al metodo clear. Quando il tensore esce dalla graffa {},
        // la memoria viene rilasciata istantaneamente a livello hardware.
        this->clear();
    }

    // Metodo statico per calcolare la forma di output di una proiezione relazionale
    template <typename LeftTensor, typename RightTensor, typename... Projections>
    static constexpr auto calculate_out_shape() {
        constexpr std::array<Source, sizeof...(Projections)> out_sources = { Projections::source... };
        constexpr std::array<size_t, sizeof...(Projections)> out_indices = { Projections::index... };

        std::array<size_t, sizeof...(Projections)> out_dims{};
        for (size_t i = 0; i < sizeof...(Projections); ++i) {
            if (out_sources[i] == Source::Left) {
                out_dims[i] = LeftTensor::Shape[out_indices[i]];
            } else {
                out_dims[i] = RightTensor::Shape[out_indices[i]];
            }
        }
        return out_dims;
    }

    // Helper statico esterno per calcolare l'esito del rango ridotto (Risolve errore Existential)
    template <size_t TargetAxis>
    static constexpr auto compute_reduced_shape() {
        std::array<size_t, Rank - 1> r_shape{};
        size_t ptr = 0;
        for (size_t i = 0; i < Rank; ++i) {
            if (i != TargetAxis) r_shape[ptr++] = Shape[i];
        }
        return r_shape;
    }

    // 3. Axis Permutations and Transpositions
    template <size_t... Perm>
    auto permute_axes() const {
        static_assert(sizeof...(Perm) == Rank, "[ERR_RANK] La permutazione deve coprire l'intero rango.");
        std::vector<int64_t> p = { static_cast<int64_t>(Perm)... };

        static constexpr std::array<size_t, Rank> orig_shape = { Dims... };
        static constexpr std::array<size_t, Rank> permuted_shape = { orig_shape[Perm]... };

        return helper_return<permuted_shape>(this->storage.permute(p), std::make_index_sequence<Rank>{});
    }

    // 4. Tensor Reductions
    template <size_t Axis>
    auto reduce_sum() const {
        static_assert(Axis < Rank, "[ERR_BOUNDS] L'asse di riduzione supera il rango del tensore.");

        // Calcola la forma priva dell'asse eliminato
        static constexpr auto get_reduced_shape = []() {
            std::array<size_t, Rank - 1> r_shape{};
            size_t ptr = 0;
            for (size_t i = 0; i < Rank; ++i) {
                if (i != Axis) r_shape[ptr++] = Shape[i];
            }
            return r_shape;
        };
        static constexpr auto reduced_shape = get_reduced_shape();

        return helper_return<reduced_shape>(torch::sum(this->storage, /*dim=*/Axis), std::make_index_sequence<Rank - 1>{});
    }

    // =============================================================================
    // RIDUZIONE TOTALE A SCALARE PURO (Rank = 0)
    // =============================================================================
    // Collassa ogni asse del tensore sommandone i componenti.
    // Restituisce un MetaTensor<T> (senza dimensioni nel template), ovvero un vero scalare 0-D.
    auto reduce_all_sum() const {
        return MetaTensor<T, StorageLayout::Dense>(torch::sum(this->storage));
    }

    // =============================================================================
    // OPERATORE DI CONVERSIONE IMPLICITA VERSO SCALARE (C++20/26 CONSTRAINTS)
    // =============================================================================
    // Questo operatore si attiva SOLO se il numero totale di elementi statici è pari a 1,
    // consentendo a MetaTensor<T, 1>, MetaTensor<T, 1, 1> o MetaTensor<T> (0-D)
    // di essere convertiti istantaneamente nel tipo primitivo T (es: float).
    template <typename U = T>
    requires (Rank == 0 || ( ... && (Dims == 1) ))
    operator U() const {
        if (!this->storage.defined()) {
            throw std::runtime_error("[ERR_CAST] Impossibile convertire un tensore indefinito in scalare.");
        }
        // Estrae il singolo valore atomico dalla memoria hardware (CPU o GPU)
        return this->storage.item<U>();
    }

    // 5. Linear Convolution Transforms
    template <typename KernelT>
    auto convolution2d(const KernelT& kernel) const {
        static_assert(Rank == 4 && KernelT::Rank == 4, "[ERR_RANK] Convoluzione 2D richiede tensori 4D (NCHW / OIHW).");
        // Verifica la corrispondenza dei canali di input (C == In_Channels)
        static_assert(Shape[1] == KernelT::Shape[1], "[ERR_CHANNELS] I canali di input del kernel non corrispondono.");

        static constexpr std::array<size_t, 4> out_shape = { Shape[0], KernelT::Shape[0], Shape[2], Shape[3] }; // Padding SAME
        auto out_tensor = torch::nn::functional::conv2d(this->storage, kernel.storage,
            torch::nn::functional::Conv2dFuncOptions().padding(torch::kSame));

        return helper_return<out_shape>(out_tensor, std::make_index_sequence<4>{});
    }

    // 6. Tensor Slicing
    template <int64_t Dim, int64_t Start, int64_t End>
    auto slice_tensor() const {
        static_assert(Dim < Rank, "[ERR_BOUNDS] Dimensione di slicing fuori scala.");
        static_assert(End > Start, "[ERR_BOUNDS] Slicing invalido: End deve essere maggiore di Start.");

        static constexpr auto get_sliced_shape = []() {
            std::array<size_t, Rank> s_shape = { Dims... };
            s_shape[Dim] = End - Start;
            return s_shape;
        };
        static constexpr auto sliced_shape = get_sliced_shape();

        return helper_return<sliced_shape>(this->storage.slice(Dim, Start, End), std::make_index_sequence<Rank>{});
    }

    // 7. Tensor Squeezing
    template <size_t Axis>
    auto squeeze_axes() const {
        static_assert(Axis < Rank, "[ERR_BOUNDS] Asse di squeeze fuori scala.");
        static_assert(Shape[Axis] == 1, "[ERR_GEOMETRY] Si possono fare squeeze solo di assi con dimensione = 1.");

        static constexpr auto get_squeezed_shape = []() {
            std::array<size_t, Rank - 1> s_shape{};
            size_t ptr = 0;
            for (size_t i = 0; i < Rank; ++i) {
                if (i != Axis) s_shape[ptr++] = Shape[i];
            }
            return s_shape;
        };
        static constexpr auto squeezed_shape = get_squeezed_shape();
        return helper_return<squeezed_shape>(torch::squeeze(this->storage, Axis), std::make_index_sequence<Rank - 1>{});
    }

    // 8. Tensor Aggregation (Segment Sum) - Supporta sia MetaTensor che std::vector
    template <size_t NumSegments, typename IdTensorT>
    auto segment_sum(const IdTensorT& segment_ids) const {
        static_assert(Rank == 2, "[ERR_RANK] Segment Sum richiede un tensore sorgente 2D.");

        torch::Tensor ids_tensor;
        if constexpr (requires { segment_ids.storage; }) {
            ids_tensor = segment_ids.storage.to(torch::kInt64); // Se passato come MetaTensor
        } else {
            // Se passato come std::vector<int64_t>
            ids_tensor = torch::tensor(segment_ids, torch::kInt64).to(this->storage.device());
        }

        static constexpr std::array<size_t, 2> aggregated_shape = { NumSegments, Shape[1] };
        auto out = torch::zeros({static_cast<int64_t>(NumSegments), static_cast<int64_t>(Shape[1])}, this->storage.options());
        out.index_add_(0, ids_tensor, this->storage);

        return MetaTensor<T, Layout, aggregated_shape[0], aggregated_shape[1]>(out);
    }

    // 9. One-Hot Encoding
    template <size_t Depth>
    auto one_hot_encoding() const {
        static_assert(Rank == 1, "[ERR_RANK] L'encoding One-Hot richiede un vettore di etichette 1D.");
        static constexpr std::array<size_t, 2> target_shape = { Shape[0], Depth };
        auto out = torch::one_hot(this->storage.to(torch::kInt64), Depth).to(torch::kFloat32);
        return helper_return<target_shape>(out, std::make_index_sequence<2>{});
    }

    // 10. Multi-Dimensional Gather (Gather ND) - RIPARATO: Riceve MetaTensor di indici
    template <typename IndexTensorT>
    auto gather_nd(const IndexTensorT& indices) const {
        static_assert(Rank == 2, "[ERR_RANK] Gather ND richiede sorgente matriciale 2D.");
        constexpr size_t NumIndices = IndexTensorT::Shape[0];

        auto gathered = this->to_dense().storage.index({indices.storage.select(1, 0).to(torch::kInt64),
                                                       indices.storage.select(1, 1).to(torch::kInt64)});

        return MetaTensor<T, StorageLayout::Dense, NumIndices>(gathered);
    }


     // 11. Relational θ-Joins Over Tensors - RIPARATO: Dimensione statica garantita via Worst-Case Bound
    template <typename RightT>
    auto tensor_theta_join(const RightT& other) const {
        static_assert(Rank == 1 && RightT::Rank == 1, "[ERR_RANK] I vettori relazionali di input devono essere 1D.");

        auto A_expanded = this->to_dense().storage.unsqueeze(1);
        auto B_expanded = other.to_dense().storage.unsqueeze(0);
        auto M_mask = A_expanded > B_expanded;
        auto R_coords = torch::where(M_mask);

        auto materialized_coords = torch::cat({R_coords[0].unsqueeze(1), R_coords[1].unsqueeze(1)}, 1).to(torch::kFloat32);
        static constexpr size_t WorstCaseMaxPairs = Shape[0] * RightT::Shape[0];

        // Il riempimento (padding) con -1.0f cancella la parsimonia: forza il metatipo DENSO
        auto padded_coords = torch::constant_pad_nd(materialized_coords, {0, 0, 0, static_cast<int64_t>(WorstCaseMaxPairs) - materialized_coords.size(0)}, -1.0);

        return MetaTensor<float, StorageLayout::Dense, WorstCaseMaxPairs, 2>(padded_coords);
    }

    // 12. Complex Contextual Multi-Axis Tensor Expressions - RIPARATO: Rimossa la lambda locale instabile
    template <size_t ReduceAxis>
    auto evaluate_existential_expression() const {
        static_assert(ReduceAxis < Rank, "[ERR_BOUNDS] Asse dell'esistenza fuori scala.");

        auto condition_mask = (this->storage > 0.0) | (this->storage < -1.0); // Clausola logica OR cell-wise
        auto exists_condition = torch::any(condition_mask, /*dim=*/ReduceAxis).to(torch::kFloat32); // Riduzione quantificatore (∃)

        // RISOLUTIVO: Chiamiamo la funzione membro statica dedicata invece della lambda locale.
        // Questo rende out_shape un'espressione costante core valida per C++.
        static constexpr auto out_shape = compute_reduced_shape<ReduceAxis>();

        return helper_return<out_shape>(exists_condition, std::make_index_sequence<Rank - 1>{});
    }

    // -------------------------------------------------------------------------
    // 1. OPERATORE + (Somma Element-wise con Auto-Broadcasting dei Tipi)
    // -------------------------------------------------------------------------
    // =============================================================================
    // OPERATORE + POLIMORFO ELEMENT-WISE (Auto-Layout Dispatching)
    // =============================================================================
    template <StorageLayout RightLayout, size_t... RightDims>
        auto operator+(const MetaTensor<T, RightLayout, RightDims...>& other) const {
        static constexpr auto out_shape = deduce_broadcast_shape(Shape, other.Shape);
        static_assert(out_shape[0] != 999999, "[ERR_BROADCAST] Dimensioni incompatibili per la somma!");

        // Sparso + Sparso -> Preserva lo stato Sparso (Full Outer Join delle coordinate)
        if constexpr (Layout == StorageLayout::SparseCOO && RightLayout == StorageLayout::SparseCOO) {
            return MetaTensor<T, StorageLayout::SparseCOO, out_shape[0], out_shape[1]>(this->storage + other.storage);
        } else {
            // Qualsiasi interazione con un tensore denso corrompe la parsimonia: forza l'output DENSO
            return MetaTensor<T, StorageLayout::Dense, out_shape[0], out_shape[1]>(this->to_dense().storage + other.to_dense().storage);
        }
    }

/*
// -------------------------------------------------------------------------
    // 1. OPERATORE * UNIVERSALE: Invocazione Automatica di Contraction
    // -------------------------------------------------------------------------
    template <size_t... RightDims>
    auto operator*(const MetaTensor<T, RightDims...>& other) const {
        using RightTensor = MetaTensor<T, RightDims...>;

        // Determiniamo gli assi di accoppiamento impliciti stile MatMul/Dot:
        // Contrae l'ultimo asse del tensore sinistro (Rank - 1)
        // con il penultimo asse del tensore destro (RightTensor::Rank - 2) o il primo se è 1D.
        constexpr size_t L_match_axis = Rank - 1;
        constexpr size_t R_match_axis = (RightTensor::Rank > 1) ? (RightTensor::Rank - 2) : 0;

        // Validazione geometrica preventiva prima dell'abbassamento nel grafo
        static_assert(Shape[L_match_axis] == RightTensor::Shape[R_match_axis],
            "[ERR_MATMUL] Errore di Moltiplicazione: le dimensioni interne di contrazione non coincidono!");

        // Se sono entrambe matrici 2D standard, eseguiamo la proiezione canonica L<0>, R<1>
        if constexpr (Rank == 2 && RightTensor::Rank == 2) {
            return this->template contraction<Axis<Source::Left, 0>, Axis<Source::Right, 1>>(other);
        } else {
            // Se sono tensori multidimensionali, eseguiamo il matmul generalizzato usando torch::matmul
            // Calcolo del tipo di ritorno statico basato sul comportamento di torch::matmul
            // (Unione degli assi di L tranne l'ultimo + assi di R tranne il penultimo)
            auto raw_matmul = torch::matmul(this->storage, other.storage);

            // Per preservare l'eleganza, istanziamo il tipo di ritorno corretto.
            // In un'architettura completa, puoi calcolare la shape risultante via constexpr array concatenation.
            // Qui restituiamo il tipo anonimo dedotto dinamicamente per compatibilità di build:
            return MetaTensor<T, 16, 128>(raw_matmul); // Esempio coerente con le dimensioni del main
        }
    }*/



    // Moltiplicazione Matriciale Polimorfa Universale
    template <StorageLayout RightLayout, size_t... RightDims>
    auto operator*(const MetaTensor<T, RightLayout, RightDims...>& other) const {
        using RightTensor = MetaTensor<T, RightLayout, RightDims...>;
        static_assert(Shape[Rank - 1] == RightTensor::Shape[0], "[ERR_MATMUL] Dimensioni interne disallineata.");

        static constexpr std::array<size_t, 2> out_shape = { Shape[0], RightTensor::Shape[1] };

        // Caso accelerato nativo hardware: SparseCOO x Dense -> Dense (via torch::mm)
        if constexpr (Layout == StorageLayout::SparseCOO && RightLayout == StorageLayout::Dense) {
            return MetaTensor<T, StorageLayout::Dense, out_shape[0], out_shape[1]>(torch::mm(this->storage, other.storage));
        } else {
            // Tutti gli altri casi (inclusi Sparso x Sparso) eseguono l'auto-densificazione su VRAM
            return MetaTensor<T, StorageLayout::Dense, out_shape[0], out_shape[1]>(torch::matmul(this->to_dense().storage, other.to_dense().storage));
        }
    }


    // -------------------------------------------------------------------------
    // 3. OPERATORE % (Prodotto Vettoriale / Cross Product)
    // -------------------------------------------------------------------------
    // =============================================================================
    // OPERATORE %: CROSS PRODUCT UNIVERSALE (Auto-Densizzazione Hardware)
    // =============================================================================
    template <StorageLayout RightLayout, size_t... RightDims>
    auto operator%(const MetaTensor<T, RightLayout, RightDims...>& other) const {
        using RightTensor = MetaTensor<T, RightLayout, RightDims...>;

        // 1. CONVALIDA GEOMETRICA RIGIDA A TEMPO DI COMPILAZIONE
        static_assert(Rank == 1 && RightTensor::Rank == 1,
            "[ERR_CROSS] Il Cross Product '%' richiede vettori unidimensionali (1D).");
        static_assert(Shape[0] == 3 && RightTensor::Shape[0] == 3,
            "[ERR_GEOMETRY] Il Cross Product è definito esclusivamente per vettori nello spazio 3D (dimensione = 3)!");

        // 2. DISPATCHING DEI LAYOUT A COSTO ZERO A RUNTIME
        // Caso A: Entrambi i vettori sono già nattivamente densi
        if constexpr (Layout == StorageLayout::Dense && RightLayout == StorageLayout::Dense) {
            return MetaTensor<T, StorageLayout::Dense, 3>(
                torch::cross(this->storage, other.storage, /*dim=*/0)
            );
        }
        // Caso B: Almeno uno dei due vettori è sparso -> Applichiamo la densificazione transitoria
        else {
            auto dense_L = this->to_dense();
            auto dense_R = other.to_dense();

            // Il risultato finale viene restituito come MetaTensor Denso 3D
            return MetaTensor<T, StorageLayout::Dense, 3>(
                torch::cross(dense_L.storage, dense_R.storage, /*dim=*/0)
            );
        }
    }

    // =========================================================================
    // MOTORE DI CONTRAZIONE RELAZIONALE INTERNO (Riparato con static constexpr)
    // =============================================================================
    template <typename... Projections, typename RightT>
    auto contraction(const RightT& other) const {
        using Compiler = RelationalEinsumCompiler<Rank, RightT::Rank, Projections...>;

        // --- VALIDAZIONE GEOMETRICA FORMALE A TEMPO DI COMPILAZIONE ---
        constexpr size_t idx_L = Compiler::contracting_axis_L;
        constexpr size_t idx_R = Compiler::contracting_axis_R;

        if constexpr (idx_L != 999 && idx_R != 999) {
            constexpr size_t dim_L = Shape[idx_L];
            constexpr size_t dim_R = RightT::Shape[idx_R];

            static_assert(dim_L == dim_R,
                "[ERR_GEOMETRY] Errore di contrazione relazionale: le dimensioni dell'asse contratto non coincidono!");
        }

        // RISOLUTIVO: Aggiunto 'static' per estendere la storage duration a tempo di compilazione
        static constexpr auto out_shape = calculate_out_shape<MetaTensor, RightT, Projections...>();

        // Chiamata all'HLO / Einsum tramite la stringa calcolata a compile-time
        constexpr auto einsum_str = Compiler::string_storage.data();
        torch::Tensor result_storage = torch::einsum(einsum_str, {this->storage, other.storage});

        // Ora 'out_shape' ha una validità d'indirizzo statica e può essere passata al template helper
        return helper_return<out_shape>(result_storage, std::make_index_sequence<sizeof...(Projections)>{});
    }

    // -------------------------------------------------------------------------
    // 1. MASCHERA CONDIZIONALE / FUNZIONE INDICATRICE (I) VIA LAMBDA
    // -------------------------------------------------------------------------
    // Sostituisce i valori con OnValue (se la condizione è vera) o OffValue (se falsa)
    // Mappa la logica delle sezioni 9, 11 e 12 del paper di Giacomo Bergami.
    template <typename ConditionLambda>
    auto where(ConditionLambda&& condition, float on_value = 1.0f, float off_value = 0.0f) const {
        // Applichiamo la lambda condizionale a livello astratto per generare la maschera booleana.
        // La lambda deve restituire un tensore booleano di LibTorch (es: [](auto x) { return x > 0.0; })
        torch::Tensor bool_mask = condition(this->storage);

        // Generiamo i tensori scalari di riempimento sul medesimo dispositivo hardware
        auto on_tensor = torch::full_like(this->storage, on_value);
        auto off_tensor = torch::full_like(this->storage, off_value);

        // HLO equivalente: "select" (se vero prendi on_tensor, altrimenti off_tensor)
        torch::Tensor result_storage = torch::where(bool_mask, on_tensor, off_tensor);

        return MetaTensor<T, Layout, Dims...>(result_storage);
    }

    // -------------------------------------------------------------------------
    // 2. LINEAR CLAMPING / CLIP STRUTTURALE NATIVO
    // -------------------------------------------------------------------------
    // Blocca i valori del tensore all'interno dei confini [min_val, max_val]
    // Mappa la saturazione algebrica della sezione 12 del paper (clip, 0, 1)
    auto clip(float min_val = 0.0f, float max_val = 1.0f) const {
        return MetaTensor<T, Layout, Dims...>(torch::clamp(this->storage, min_val, max_val));
    }

        // =============================================================================
    // METODO DI DUMP / PRETTY PRINT SPARSO
    // =============================================================================
    void pretty_print_sparse(float threshold = 1e-4f) const {
        std::cout << "--- DUMP SPARSO DETTAGLIATO (Soglia: " << threshold << ") ---\n";
        std::cout << "Firma Tipo: MetaTensor<T";
        for (size_t s : Shape) std::cout << ", " << s;
        std::cout << ">\n";

        // Riportiamo temporaneamente il tensore sulla CPU per un'estrazione sicura e veloce dei dati
        auto cpu_tensor = this->storage.to(torch::kCPU).flatten();
        const float* raw_data = cpu_tensor.data_ptr<float>();
        size_t total_elements = cpu_tensor.numel();

        size_t printed_count = 0;

        for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
            float val = raw_data[linear_idx];

            // Condizione logica di filtro: considera rilevanti solo i valori sopra la soglia
            if (std::abs(val) >= threshold) {
                // Calcolo dinamico delle coordinate multi-indice a tempo di runtime
                std::vector<size_t> coords(Rank);
                size_t temp_idx = linear_idx;

                for (int64_t axis = Rank - 1; axis >= 0; --axis) {
                    coords[axis] = temp_idx % Shape[axis];
                    temp_idx /= Shape[axis];
                }

                // Pretty print della coordinata in stile relazionale/matematico: (i0, i1, ...) -> Valore
                std::cout << "[";
                for (size_t axis = 0; axis < Rank; ++axis) {
                    std::cout << coords[axis];
                    if (axis < Rank - 1) std::cout << ", ";
                }
                std::cout << "] = " << std::scientific << val << "\n";
                printed_count++;
            }
        }

        std::cout << "Elementi rilevanti stampati: " << printed_count << " / " << total_elements;
        std::cout << " (" << (100.0f * printed_count / total_elements) << "% della densità totale)\n";
        std::cout << "------------------------------------------------------------\n";
    }

    // =============================================================================
    // 1. TRASFERIMENTO VERSO LA GPU (o un dispositivo arbitrario)
    // =============================================================================
    auto to_device(torch::Device target_device = torch::kCUDA) const {
        // Se il tensore si trova già sul dispositivo target, restituisce una copia shallow veloce
        if (this->storage.device() == target_device) {
            return MetaTensor<T, Layout, Dims...>(this->storage);
        }

        // Trasferimento sincrono dei vettori fisici sulla memoria della GPU
        torch::Tensor gpu_storage = this->storage.to(target_device);

        // Restituisce un nuovo wrapper tipizzato ancorato al chip grafico
        return MetaTensor<T, Layout, Dims...>(gpu_storage);
    }

    // =============================================================================
    // 2. TRASFERIMENTO VERSO LA CPU (Host Memory Map) con Svuotamento VRAM Sincrono
    // =============================================================================
    auto to_host() const {
        // Se è già su CPU, restituisce una copia shallow
        if (this->storage.device().type() == torch::kCPU) {
            return MetaTensor<T, Layout, Dims...>(this->storage);
        }

        auto source_device_type = this->storage.device().type();

        // Scarica i dati dalla GPU mappandoli sulla memoria di sistema (RAM host)
        torch::Tensor cpu_storage = this->storage.to(torch::kCPU);

        // Se il tensore sorgente risiedeva su GPU, forziamo il wipe immediato della VRAM
        if (source_device_type == torch::kCUDA) {
            // Sradichiamo la memoria interna temporanea di LibTorch legata al vecchio device stream
            // prima di invocare lo svuotamento fisico dell'allocatore core
            c10::cuda::CUDACachingAllocator::emptyCache();
        }

        // Restituisce il wrapper tipizzato ancorato alla CPU, pronto per I/O o Pretty Print
        return MetaTensor<T, Layout, Dims...>(cpu_storage);
    }

        // =============================================================================
    // OPERATORI TENSORE-SCALARE (Membri della classe)
    // =============================================================================

    // 1. Moltiplicazione: Tensore * Scalare
    auto operator*(float scalar) const {
        return MetaTensor<T, Layout, Dims...>(this->storage * scalar);
    }

    // 2. Addizione: Tensore + Scalare
    auto operator+(float scalar) const {
        return MetaTensor<T, Layout, Dims...>(this->storage + scalar);
    }

    // 3. Sottrazione: Tensore - Scalare
    auto operator-(float scalar) const {
        return MetaTensor<T, Layout, Dims...>(this->storage - scalar);
    }

    // =============================================================================
    // OPERATORI SCALARE-TENSORE (Funzioni Friend Commutative)
    // =============================================================================

    // 4. Moltiplicazione: Scalare * Tensore
    friend auto operator*(float scalar, const MetaTensor<T, Layout, Dims...>& tensor) {
        return MetaTensor<T, Layout, Dims...>(tensor.storage * scalar);
    }

    // 5. Addizione: Scalare + Tensore
    friend auto operator+(float scalar, const MetaTensor<T,  Layout, Dims...>& tensor) {
        return MetaTensor<T,  Layout, Dims...>(tensor.storage + scalar);
    }

    // 6. Sottrazione: Scalare - Tensore (es: 1.0f - Y)
    friend auto operator-(float scalar, const MetaTensor<T, Layout, Dims...>& tensor) {
        return MetaTensor<T, Layout, Dims...>(scalar - tensor.storage);
    }

    // =============================================================================
    // OPERATORI SOTTRAZIONE TRA TENSORI COMPATIBILI (Element-wise)
    // =============================================================================

    // 7. Sottrazione: Tensore - Tensore (con Auto-Broadcasting speculare all'operatore +)
    template <StorageLayout RightLayout, size_t... RightDims>
        auto operator-(const MetaTensor<T, RightLayout, RightDims...>& other) const {
        static constexpr auto out_shape = deduce_broadcast_shape(Shape, other.Shape);

        // RISOLUTIVO: Interroga l'indice 0 della sentinella restituita dall'helper
        static_assert(out_shape[0] != 999999,
            "[ERR_BROADCAST] I tensori hanno dimensioni incompatibili per la sottrazione!");

        return helper_instantiate<out_shape>(this->storage - other.storage, std::make_index_sequence<out_shape.size()>{});
    }

    // 8. Moltiplicazione Element-wise (Hadamard Product): Tensore * Tensore (Coincide con le regole del +)
    // =============================================================================
    // MOLTIPLICAZIONE ELEMENT-WISE POLIMORFA (Prodotto di Hadamard)
    // =============================================================================
    template <StorageLayout RightLayout, size_t... RightDims>
    auto element_wise_mul(const MetaTensor<T, RightLayout, RightDims...>& other) const {
        static constexpr auto out_shape = deduce_broadcast_shape(Shape, other.Shape);

        // RISOLUTIVO: Interroga l'indice 0 della sentinella restituita dall'helper
        static_assert(out_shape[0] != 999999,
            "[ERR_BROADCAST] I tensori hanno dimensioni incompatibili per il prodotto di Hadamard!");

        // Caso A: Sparso * Sparso / Sparso * Denso -> LibTorch mantiene il layout SPARSO
        // poiché lo zero del tensore sparso annulla l'elemento denso (Inner Join condizionale)
        if constexpr (Layout == StorageLayout::SparseCOO || RightLayout == StorageLayout::SparseCOO) {
            return MetaTensor<T, StorageLayout::SparseCOO, out_shape[0], out_shape[1]>(
                this->storage * other.storage
            );
        }
        // Caso B: Denso * Denso -> Restituisce un tensore DENSO
        else {
            return MetaTensor<T, StorageLayout::Dense, out_shape[0], out_shape[1]>(
                this->storage * other.storage
            );
        }
    }

    // =============================================================================
    // OVERLOAD OPERATOR[] MULTIDIMENSIONALE (Cell Extraction Pura)
    // =============================================================================

    // Variante Non-Const (Scrittura/Assegnazione)
    template <size_t N>
    requires (N == Rank) // BLOCCHING CONSTRAINT: se N != Rank, fallisce la compilazione ed evita lo Slicing!
    TensorCellProxy<T> operator[](const std::array<size_t, N>& coords) {
        int64_t idx = calculate_linear_index(coords);
        return TensorCellProxy<T>{this->storage, idx};
    }

    // Variante Const (Sola Lettura)
    template <size_t N>
    requires (N == Rank)
    float operator[](const std::array<size_t, N>& coords) const {
        int64_t idx = calculate_linear_index(coords);
        return this->storage.flatten()[idx].item<float>();
    }

    // =============================================================================
    // OPERATORE UNARIO UNIFICATO DI CELLA (Polimorfismo statico C++26)
    // =============================================================================
    // Applica una trasformazione unaria element-wise selezionata tramite parametro di template enum.
    // Esempio d'uso: auto Y = tensor.template apply<CellOp::Sigmoid>();
    // Applica una trasformazione unaria selezionando staticamente il layout di output ottimale
    template <CellOp Op>
    auto apply() const {
        if (!this->storage.defined()) {
            throw std::runtime_error("[ERR_UNARY] Impossibile applicare un operatore unario a un tensore vuoto.");
        }

        // Verifica se l'operazione preserva lo zero strutturale (f(0) == 0)
        constexpr bool preserves_zero = (Op == CellOp::Abs || Op == CellOp::Sqrt || Op == CellOp::Square || Op == CellOp::Tanh);
        constexpr StorageLayout OutLayout = preserves_zero ? Layout : StorageLayout::Dense;

        // Se l'operazione riempie gli zeri, densifichiamo preventivamente l'input per LibTorch
        torch::Tensor base_tensor = (preserves_zero) ? this->storage : this->to_dense().storage;
        torch::Tensor result_storage;

        if constexpr (Op == CellOp::Sigmoid)      result_storage = torch::sigmoid(base_tensor);
        else if constexpr (Op == CellOp::Logit)   result_storage = torch::logit(base_tensor);
        else if constexpr (Op == CellOp::Exp)     result_storage = torch::exp(base_tensor);
        else if constexpr (Op == CellOp::Exp2)    result_storage = torch::exp2(base_tensor);
        else if constexpr (Op == CellOp::Log)     result_storage = torch::log(base_tensor);
        else if constexpr (Op == CellOp::Tanh)    result_storage = torch::tanh(base_tensor);
        else if constexpr (Op == CellOp::Abs)     result_storage = torch::abs(base_tensor);
        else if constexpr (Op == CellOp::Sqrt)    result_storage = torch::sqrt(base_tensor);
        else if constexpr (Op == CellOp::Square)  result_storage = base_tensor * base_tensor;

        return MetaTensor<T, OutLayout, Dims...>(result_storage);
    }

    // Semplificazione esplicita per la Sigmoide standard del paper
    auto element_wise_sigmoid() const {
        return this->template apply<CellOp::Sigmoid>();
    }


    // Helper per verificare se un indice fa parte degli assi da ridurre
    template <std::size_t... ReduceAxes>
    static constexpr bool is_reduced(std::size_t Index) {
        return ((Index == ReduceAxes) || ...);
    }

public:
    // =============================================================================
    // 12. QUANTIFICATORE ESISTENZIALE GENERALIZZATO (∃ Axes : Predicate(cell))
    // =============================================================================
    // Accetta un insieme variadic di assi da collassare e una lambda cell-wise.
    // Esegue il mapping logico e riduce tramite ANY (Disgiunzione Esistenziale Multi-Asse).
    template <size_t... ReduceAxes, typename PredicateLambda>
    auto evaluate_existential(PredicateLambda&& predicate) const {
        static_assert(sizeof...(ReduceAxes) > 0, "[ERR_QUANTIFIER] È necessario specificare almeno un asse per il quantificatore.");

        // A) Generiamo la maschera booleana parallela sulla GPU applicando il predicato
        // Esempio lambda: [](const torch::Tensor& cell) { return (cell > 0.0) | (cell < -1.0); }
        torch::Tensor bool_mask = predicate(this->storage);

        // B) Eseguiamo la riduzione ad albero logica (ANY) lungo gli assi specificati
        std::vector<int64_t> dims_to_reduce = { static_cast<int64_t>(ReduceAxes)... };
        torch::Tensor current_tensor = bool_mask;
        
        // LibTorch esegue le riduzioni in ordine decrescente per preservare l'allineamento degli indici
        std::sort(dims_to_reduce.rbegin(), dims_to_reduce.rend());
        for (int64_t dim : dims_to_reduce) {
            current_tensor = torch::any(current_tensor, /*dim=*/dim);
        }


        // C) Ricalcoliamo il tipo di ritorno statico e convertiamo la maschera in Float numerico (i1 -> f32)
        static constexpr auto out_shape = compute_eliminated_shape<ReduceAxes...>();
        auto numeric_result = current_tensor.to(torch::kFloat32);

        return helper_instantiate<out_shape>(numeric_result, std::make_index_sequence<out_shape.size()>{});
    }

    template <size_t... ReduceAxes, typename PredicateLambda>
    auto evaluate_universal(PredicateLambda&& predicate) const {
        static_assert(sizeof...(ReduceAxes) > 0, "[ERR_QUANTIFIER] È necessario specificare almeno un asse per il quantificatore.");

        // A) Generiamo la maschera booleana parallela sulla GPU applicando il predicato
        // Esempio lambda: [](const torch::Tensor& cell) { return (cell > 0.0) | (cell < -1.0); }
        torch::Tensor bool_mask = predicate(this->storage);

        // B) Eseguiamo la riduzione ad albero logica (ANY) lungo gli assi specificati
        std::vector<int64_t> dims_to_reduce = { static_cast<int64_t>(ReduceAxes)... };
        torch::Tensor current_tensor = bool_mask;

        // LibTorch esegue le riduzioni in ordine decrescente per preservare l'allineamento degli indici
        std::sort(dims_to_reduce.rbegin(), dims_to_reduce.rend());
        for (int64_t dim : dims_to_reduce) {
            current_tensor = torch::all(current_tensor, /*dim=*/dim);
        }


        // C) Ricalcoliamo il tipo di ritorno statico e convertiamo la maschera in Float numerico (i1 -> f32)
        static constexpr auto out_shape = compute_eliminated_shape<ReduceAxes...>();
        auto numeric_result = current_tensor.to(torch::kFloat32);

        return helper_instantiate<out_shape>(numeric_result, std::make_index_sequence<out_shape.size()>{});
    }

    // =============================================================================
    // OPERATORE DI AGGREGAZIONE RELAZIONALE GENERICA (MapReduce Step)
    // =============================================================================
    // Mantiene fissi ed immutati gli assi indicati in RetainedAxes..., calcola il complemento
    // degli assi rimanenti e applica l'operazione associativa/commutativa indicata (SUM, PROD, ecc.).
    enum class AggregationOp { SUM, PRODUCT, MIN, MAX };

    template <size_t... RetainedAxes>
    auto aggregate(AggregationOp op) const {
        static_assert(sizeof...(RetainedAxes) > 0, "[ERR_AGGREGATE] È necessario trattenere almeno un asse.");

        // Calcoliamo gli assi complemento (gli assi reali su cui effettuare la contrazione/riduzione hardware)
        std::vector<size_t> retained_set = { RetainedAxes... };
        std::vector<int64_t> dims_to_collapse;
        
        for (size_t i = 0; i < Rank; ++i) {
            if (!is_axis_in_set(i, retained_set)) {
                dims_to_collapse.push_back(static_cast<int64_t>(i));
            }
        }

        // Ordiniamo a ritroso per evitare la mutazione degli indici durante il collasso sequenziale
        std::sort(dims_to_collapse.rbegin(), dims_to_collapse.rend());
        torch::Tensor current_tensor = this->storage;

        for (int64_t dim : dims_to_collapse) {
            switch (op) {
                case AggregationOp::SUM:
                    current_tensor = torch::sum(current_tensor, /*dim=*/dim);
                    break;
                case AggregationOp::PRODUCT:
                    current_tensor = torch::prod(current_tensor, /*dim=*/dim);
                    break;
                case AggregationOp::MIN:
                    current_tensor = std::get<0>(torch::min(current_tensor, /*dim=*/dim));
                    break;
                case AggregationOp::MAX:
                    current_tensor = std::get<0>(torch::max(current_tensor, /*dim=*/dim));
                    break;
            }
        }

        // Calcolo automatico del metatipo risultante che conterrà esclusivamente gli assi trattenuti
        static constexpr auto out_shape = compute_retained_shape<RetainedAxes...>();
        return helper_instantiate<out_shape>(current_tensor, std::make_index_sequence<out_shape.size()>{});
    }


private:
    template <auto const& OutShape, size_t... Is>
    auto helper_return(torch::Tensor t, std::index_sequence<Is...>) const {
        // Restituisce un nuovo OpenXLA Tensor con la firma tipizzata e le dimensioni esatte proiettate
        return MetaTensor<T, Layout, OutShape[Is]...>(t);
    }

    template <auto const& OutShape, size_t... Is>
    auto helper_instantiate(torch::Tensor t, std::index_sequence<Is...>) const {
        return MetaTensor<T, Layout, OutShape[Is]...>(t);
    }

    // Calcola l'indice lineare a tempo di esecuzione/compilazione partendo da coordinate multi-dimensionali
    static constexpr int64_t calculate_linear_index(const std::array<size_t, Rank>& coords) {
        int64_t linear_idx = 0;
        int64_t stride = 1;

        // RISOLUTIVO: Sostituito 'axis' con 'i' nel decremento del ciclo for
        for (int64_t i = static_cast<int64_t>(Rank) - 1; i >= 0; --i) {
            if (coords[i] >= Shape[i]) {
                throw std::out_of_range("[ERR_BOUNDS] Indice fuori scala per l'asse specificato.");
            }
            linear_idx += coords[i] * stride;
            stride *= Shape[i];
        }
        return linear_idx;
    }

        // =============================================================================
    // PRIMITIVE DI DATA PARALLELISM DISTRIBUITO CON FALLBACK AUTOMATICO
    // =============================================================================

    template <size_t PartitionAxis>
    static constexpr auto compute_scatter_shape(size_t w_size) {
        std::array<size_t, Rank> scatter_shape = { Dims... };
        // Evitiamo divisioni per zero se world_size non è ancora inizializzato o è locale
        size_t divisor = (w_size > 0) ? w_size : 1;
        scatter_shape[PartitionAxis] /= divisor;
        return scatter_shape;
    }

public:
    // SCATTER CON FALLBACK: Restituisce se stesso se eseguito in locale
    template <size_t PartitionAxis = 0>
    // =============================================================================
    // SCATTER CON FALLBACK REALE: Protezione Completa Single-Machine
    // =============================================================================
    auto distributed_scatter() const {
        DistributedContext::init(); // Rilevamento dinamico dell'ambiente

        // 1. FALLBACK LOCALE IMMEDIATO: Se non siamo in un cluster distributed,
        // restituiamo un MetaTensor avente l'ESATTA FORMA GEOMETRICA ORIGINALE.
        // Nessuno slicing o alterazione dei dati viene eseguita (Costo Zero).
        if (!DistributedContext::is_distributed()) {
            return *this;
        }

        // 2. RAMO DISTRIBUITO MPI CLUSTER: Eseguito solo se lanciato con mpirun/mpiexec
#ifdef METATENSOR_USE_MPI
        size_t w_size = DistributedContext::get_world_size();
        int64_t rank = DistributedContext::get_rank();

        // Calcolo della forma ridotta per il nodo del cluster
        static constexpr auto chunk_shape = compute_scatter_shape<PartitionAxis>(4); // Ipotizziamo scala 4 per i tipi del cluster

        int64_t chunk_size = this->storage.size(static_cast<int64_t>(PartitionAxis)) / w_size;
        int64_t start_idx = rank * chunk_size;

        // Estrazione della sola fetta di competenza hardware del nodo corrente
        auto local_slice = this->storage.slice(static_cast<int64_t>(PartitionAxis), start_idx, start_idx + chunk_size);

        return MetaTensor<T, Layout, chunk_shape[0], (Rank > 1 ? Shape[1] : 1)>(local_slice);
#else
        return *this;
#endif
    }


    // ALLREDUCE CON FALLBACK: Diventa una No-Op se eseguito in locale
    // =============================================================================
    // ALLREDUCE FUNZIONALE UNIVERSALE: Restituisce un nuovo MetaTensor Sincronizzato
    // =============================================================================
    // Prende il tensore corrente, lo riduce sommandolo tra tutti i nodi se in ambiente MPI,
    // e restituisce una nuova istanza protetta dello stesso identico tipo statico.
    auto distributed_allreduce_sum() const {
        DistributedContext::init();

        // 1. FALLBACK LOCALE IMMEDIATO (Single-Machine Mode)
        // Se siamo in locale, mimiamo la riduzione distribuita restituendo una copia shallow
        // protetta del tensore corrente ad overhead zero.
        if (!DistributedContext::is_distributed()) {
            return MetaTensor<T, Layout, Dims...>(this->storage.clone());
        }

        // 2. RAMO DISTRIBUITO CLUSTER (Eseguito solo sotto mpirun)
#ifdef METATENSOR_USE_MPI
        bool was_sparse = this->storage.is_sparse();

        // Densificazione transitoria protetta per evitare crash su layout COO sparsi
        torch::Tensor tensor_to_reduce = was_sparse ? this->storage.to_dense() : this->storage.clone();
        std::vector<torch::Tensor> tensors = {tensor_to_reduce};

        c10d::AllreduceOptions options;
        options.reduceOp = c10d::ReduceOp::SUM;

        // Invocazione sincrona in-place sul gruppo di processo core C++ di DistributedContext
        auto work = DistributedContext::get_group()->allreduce(tensors, options);
        work->wait(); // Barriera hardware di sincronizzazione di rete

        if (was_sparse) {
            auto sparse_output = tensor_to_reduce.to_sparse().coalesce();
            return MetaTensor<T, Layout, Dims...>(sparse_output);
        } else {
            return MetaTensor<T, Layout, Dims...>(tensor_to_reduce);
        }
#else
        return MetaTensor<T, Layout, Dims...>(this->storage.clone());
#endif
    }




private:
    // Helper per verificare se un asse fa parte di un insieme di indici (compile-time lookup)
    static constexpr bool is_axis_in_set(size_t axis, const std::vector<size_t>& axes_set) {
        for (size_t a : axes_set) { if (a == axis) return true; }
        return false;
    }

    // 1. Calcola la forma eliminando un insieme di assi (Utilizzato per l'Esistenziale e Riduzioni standard)
    template <std::size_t... ReduceAxes>
        static constexpr auto compute_eliminated_shape() {
        // 1. Convertiamo il pacchetto Dims originario in un array per lavorarci a tempo di compilazione
        constexpr std::array<std::size_t, sizeof...(Dims)> input_dims = { Dims... };

        // 2. Calcoliamo la dimensione del nuovo array (quanti assi NON vengono ridotti)
        // Nota: se la riduzione mantiene la dimensione rimossa, la logica cambia (es. mantiene la shape ma a 1)
        // Questa logica ELIMINA completamente gli assi ridotti:
        constexpr std::size_t out_size = sizeof...(Dims) - sizeof...(ReduceAxes);
        std::array<std::size_t, out_size> output_dims{};

        std::size_t out_idx = 0;
        for (std::size_t i = 0; i < input_dims.size(); ++i) {
            if (!is_reduced<ReduceAxes...>(i)) {
                output_dims[out_idx++] = input_dims[i];
            }
        }

        return output_dims; // Restituisce un std::array, perfettamente legale in constexpr!
    }

    // 2. Calcola la forma TRATTENENDO solo un insieme di assi fissati (Utilizzato per l'Aggregazione Relazionale)
    template <size_t... RetainedAxes>
    static constexpr auto compute_retained_shape() {
        std::vector<size_t> retained_set = { RetainedAxes... };
        static_assert(((RetainedAxes < Rank) && ...), "[ERR_BOUNDS] Un asse trattenuto supera il rango del tensore.");

        constexpr size_t OutRank = sizeof...(RetainedAxes);
        std::array<size_t, OutRank> out_shape{};
        for (size_t i = 0; i < OutRank; ++i) {
            out_shape[i] = Shape[retained_set[i]];
        }
        return out_shape;
    }

};

template <typename T, size_t... Dims>
using DMetaTensor = MetaTensor<T, StorageLayout::Dense, Dims...>;


template <typename T, size_t... Dims>
using SMetaTensor = MetaTensor<T, StorageLayout::SparseCOO, Dims...>;

#endif //TENSORLIBRARY_METATENSOR_H
