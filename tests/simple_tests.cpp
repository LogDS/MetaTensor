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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <logds/metatensor/BroadcastShape.h>
#include <logds/metatensor/GradientTape.h>
#include <logds/metatensor/RelationalEinsumCompiler.h>
#include <logds/metatensor/MetaTensor.h>
#include <string_view>

// Helper per confrontare stringhe constexpr a tempo di compilazione
constexpr bool strings_are_equal(std::string_view a, std::string_view b) {
    return a == b;
}

// =============================================================================
// 1. UNIT TEST A TEMPO DI COMPILAZIONE (METAPROGRAMMAZIONE & IR)
// =============================================================================

TEST_CASE("Metaprogramming: Validazione Constexpr dei Compilatori", "[compile-time]") {
    
    SECTION("Deduzione ed Auto-Broadcasting delle Forme Geometriche") {
        constexpr std::array<size_t, 2> lhs_shape = {1, 32};
        constexpr std::array<size_t, 2> rhs_shape = {16, 1};
        
        // Verifica formale delle regole stile NumPy/OpenXLA
        constexpr auto out_shape = deduce_broadcast_shape(lhs_shape, rhs_shape);
        
        STATIC_REQUIRE(out_shape[0] == 16);
        STATIC_REQUIRE(out_shape[1] == 32);
        
        // Verifica dello scenario fallimentare (dimensioni incompatibili)
        constexpr std::array<size_t, 2> bad_rhs = {16, 3};
        constexpr auto bad_shape = deduce_broadcast_shape(lhs_shape, bad_rhs);
        STATIC_REQUIRE(bad_shape[0] == 999999); // Sentinella d'errore
    }

    SECTION("Generazione Stringhe HLO e Riconoscimento Assi di Contrazione") {
        // Mappatura relazionale: L<0>, L<1>, R<0> su RankA=3 e RankB=2
        // Equivalente alla sommatoria di Einstein: "abc,dc->abd"
        using Compiler = RelationalEinsumCompiler<3, 2, L<0>, L<1>, R<0>>;
        
        constexpr std::string_view generated_str(Compiler::string_storage.data());
        STATIC_REQUIRE(strings_are_equal(generated_str, "abc,dc->abd"));
        
        // Verifica che l'asse contratto sia stato identificato correttamente sulle dimensioni interne
        STATIC_REQUIRE(Compiler::contracting_axis_L == 2); // Asse 'c' (posizione 2 in A)
        STATIC_REQUIRE(Compiler::contracting_axis_R == 1); // Asse 'c' (posizione 1 in B)
    }
}

// =============================================================================
// 2. UNIT TEST DI RUNTIME (ALGEBRA TENSERIALE RELAZIONALE)
// =============================================================================

TEST_CASE("Runtime: Operazioni Algebriche e Relazionali Core", "[runtime]") {
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    SECTION("Iniezione del Prodotto Matriciale Universale (Operatore *)") {
        MetaTensor<float, 4, 8> A(device);
        MetaTensor<float, 8, 2> B(device);
        
        // Riempimento deterministico per validare l'esito matematico
        A.storage = torch::ones({4, 8}, torch::device(device)) * 2.0f;
        B.storage = torch::ones({8, 2}, torch::device(device)) * 0.5f;

        // Esecuzione tramite l'operatore '*' che delega a contraction
        auto C = A * B;

        // Convalida statica del tipo di ritorno
        STATIC_REQUIRE(decltype(C)::Rank == 2);
        STATIC_REQUIRE(decltype(C)::Shape[0] == 4);
        STATIC_REQUIRE(decltype(C)::Shape[1] == 2);

        // Convalida matematica del calcolo: ogni cella deve valere (2.0 * 0.5) * 8 = 8.0f
        auto cpu_storage = C.storage.to(torch::kCPU);
        float* raw_ptr = cpu_storage.data_ptr<float>();
        
        REQUIRE_THAT(raw_ptr[0], Catch::Matchers::WithinAbs(8.0f, 1e-5f));
        REQUIRE_THAT(raw_ptr[3], Catch::Matchers::WithinAbs(8.0f, 1e-5f));
    }

    SECTION("Mappatura Relazionale: Theta-Join e Materializzazione") {
        MetaTensor<float, 4> vec_A(device);
        MetaTensor<float, 4> vec_B(device);

        // Forziamo i valori per controllare la condizione (A_i > B_j)
        vec_A.storage = torch::tensor({1.0f, 5.0f, 2.0f, 0.0f}, torch::device(device));
        vec_B.storage = torch::tensor({3.0f, 2.0f, 6.0f, 1.0f}, torch::device(device));

        // Esecuzione del Theta-Join
        // Matrice combinazioni teorica: WorstCaseMaxPairs = 4 * 4 = 16
        auto join_result = vec_A.tensor_theta_join(vec_B);

        STATIC_REQUIRE(decltype(join_result)::Rank == 2);
        STATIC_REQUIRE(decltype(join_result)::Shape[0] == 16); // Allineato al Worst-Case Bound statico
        STATIC_REQUIRE(decltype(join_result)::Shape[1] == 2);

        // Estrazione per la convalida dei record validi
        auto cpu_join = join_result.storage.to(torch::kCPU);
        float* raw_join = cpu_join.data_ptr<float>();

        // Verifichiamo la prima tupla valida generata: vec_A[1] (5.0) > vec_B[0] (3.0) -> Coordinata reale [1, 0]
        // Nota: i vettori dinamici non accoppiati subiscono il padding a -1.0
        bool found_valid_pair = false;
        for (size_t i = 0; i < 16; ++i) {
            if (raw_join[i * 2] == 1.0f && raw_join[i * 2 + 1] == 0.0f) {
                found_valid_pair = true;
                break;
            }
        }
        REQUIRE(found_valid_pair);
    }

    SECTION("Quantificatori Esistenziali Multi-Asse ed Espressioni Condizionali") {
        MetaTensor<float, 2, 4> Z(device);
        // Impostiamo celle specifiche fuori dai confini logici della disgiunzione (Z > 0 o Z < -1)
        Z.storage = torch::tensor({{-0.5f,  2.0f, -0.2f, -0.1f}, 
                                   {-0.8f, -0.9f, -0.3f, -0.4f}}, torch::device(device));

        // Riduzione lungo l'asse delle colonne (Asse 1), verificando la presenza di attivazioni (∃)
        auto existential = Z.evaluate_existential_expression<1>();

        STATIC_REQUIRE(decltype(existential)::Rank == 1);
        STATIC_REQUIRE(decltype(existential)::Shape[0] == 2); // Il rango collassa da 2D a 1D

        auto cpu_exist = existential.storage.to(torch::kCPU);
        float* raw_exist = cpu_exist.data_ptr<float>();

        // Riga 0 ha un elemento > 0 (2.0f) -> Attivata (1.0f)
        REQUIRE_THAT(raw_exist[0], Catch::Matchers::WithinAbs(1.0f, 1e-5f));
        // Riga 1 non ha elementi attivanti -> Disattivata (0.0f)
        REQUIRE_THAT(raw_exist[1], Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    }
}

// =============================================================================
// 3. UNIT TEST DEL FLUSSO DI BACKWARD (GRADIENT TAPE MULTI-TENSORE)
// =============================================================================

TEST_CASE("Runtime: Flusso dei Gradienti e Ottimizzazione SGD", "[autograd]") {
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    SECTION("Tracciamento del Nastro ed Estrazione delle Tuple dei Gradienti") {
        MetaTensor<float, 4, 2> X(device);
        MetaTensor<float, 2, 1> W(device);
        MetaTensor<float, 4, 1> b(device);
        MetaTensor<float, 4, 1> Y_true(device);

        X.storage = torch::ones({4, 2}, torch::device(device));
        W.storage = torch::ones({2, 1}, torch::device(device)) * 0.5f;
        b.storage = torch::zeros({4, 1}, torch::device(device));
        Y_true.storage = torch::ones({4, 1}, torch::device(device)) * 2.0f;

        // Registrazione dei parametri da ottimizzare
        W.watch();
        b.watch();

        MetaTensor<float, 1, 1> loss_wrapper(device);

        {
            GradientTape tape; // Avvio del nastro

            // Forward Pass: (X * W) + b -> Prodotto matriciale universale e somma broadcast
            auto Y_pred = (X * W) + b;
            
            // Calcolo Loss MSE elementare fusa ridotta a scalare
            // Nuova sintassi C++26 pulita e convalidata dai tipi statici
            auto error = Y_pred - Y_true;                        // Sfrutta l'operatore - element-wise [1]
            auto square_error = error.element_wise_mul(error);    // Sfrutta il prodotto di Hadamard element-wise [1]

            // 3. RISOLUTIVO: Collassiamo la matrice 4x1 a uno scalare puro (0-D)
            // riducendo e sommando lungo l'asse 0. Il tipo diventa MetaTensor<float> (Rank = 0)
            // CORREZIONE: Collassiamo l'intera matrice 4x1 in uno scalare puro 0-D
            // loss_scalar diventa un tipo MetaTensor<float> con Rank == 0
            auto loss_scalar = square_error.reduce_all_sum();

            // 3. GRAZIE ALL'OPERATORE DI CAST IMPLICITO:
            // loss_tensor (che è MetaTensor<float, 1>) si converte automaticamente in float!
            float host_loss = loss_scalar;

            std::cout << "Loss dell'epoca corrente: " << host_loss << "\n";

            // Utilizzo diretto all'interno dei costrutti logici di arresto o test di Catch2
            if (host_loss < 1e-4f) {
                std::cout << "Convergenza raggiunta.\n";
            }

            // Chiamata variadic multi-tensore per estrarre la tupla statica dei gradienti
            auto [dW, db] = tape.gradients(loss_scalar, W, b);

            // Convalida formale dei tipi delle matrici Jacobiane estratte
            // CONVALIDA FORMALE DEI TIPI DELLE MATRICI JACOBIANE ESTRATTE
            // RISOLUTIVO: dW deve avere la stessa identica forma dei pesi W, ovvero [2, 1]
            STATIC_REQUIRE(decltype(dW)::Rank == 2);
            STATIC_REQUIRE(decltype(dW)::Shape[0] == 2); // Corretto: prima dimensione pari a 2
            STATIC_REQUIRE(decltype(dW)::Shape[1] == 1); // Corretto: seconda dimensione pari a 1

            // db deve avere la stessa forma del bias b, ovvero [4, 1]
            STATIC_REQUIRE(decltype(db)::Rank == 2);
            STATIC_REQUIRE(decltype(db)::Shape[0] == 4);
            STATIC_REQUIRE(decltype(db)::Shape[1] == 1);

            // Verifichiamo che i gradienti fisici siano stati estratti dall'Autograd
            REQUIRE(dW.storage.defined());
            REQUIRE(db.storage.defined());

        }
    }
}


TEST_CASE("Runtime: Cell Extraction Multidimensionale via std::array", "[operators]") {
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Alloca un tensore 3D statico: MetaTensor<float, 2, 3, 4> (24 elementi totali)
    MetaTensor<float, 2, 3, 4> T(device, InitPattern::Zeros);

    SECTION("Assegnazione ed Estrazione di Celle Singole") {
        // Definiamo le coordinate esatte corrispondenti al Rango 3
        std::array<size_t, 3> coord_A = {0, 2, 1};
        std::array<size_t, 3> coord_B = {1, 1, 3};

        // Assegnazione in-place sulla GPU
        T[coord_A] = 88.5f;
        T[coord_B] = -12.0f;

        // Estrazione e verifica matematica
        float val_A = T[coord_A];
        float val_B = T[coord_B];

        REQUIRE_THAT(val_A, Catch::Matchers::WithinAbs(88.5f, 1e-5f));
        REQUIRE_THAT(val_B, Catch::Matchers::WithinAbs(-12.0f, 1e-5f));
    }

    SECTION("Protezione contro Indici Fuori Confine (Out of Range)") {
        std::array<size_t, 3> bad_coord = {0, 5, 1}; // L'asse 1 ha dimensione massima 3 (indici validi 0..2)
        REQUIRE_THROWS_AS(T[bad_coord] = 1.0f, std::out_of_range);
    }
}

#include <torch/torch.h>
#include <logds/metatensor/MetaTensor.h>
#include <logds/metatensor/Init.h>
#include <iostream>

TEST_CASE("Validazione Operatori Generalizzati", "[ex+agg]")  {
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "=== Validazione Operatori Generalizzati C++26 ===\n\n";

    // Un tensore 4D di partenza: [Batch=16, Rows=32, Cols=64, Channels=3]
    MetaTensor<float, 16, 32, 64, 3> Z( device, InitPattern::RandomNormal);

    // =========================================================================
    // TEST 1: Quantificatore Esistenziale Multi-Asse Generalizzato (∃)
    // =========================================================================
    // Riduciamo ed eliminiamo contemporaneamente l'asse 1 (32) e l'asse 2 (64)
    // applicando un predicato cellulare condizionale arbitrario via Lambda.
    // Tipo atteso dedotto dal compilatore: MetaTensor<float, 16, 3> (Rank = 2)
    auto exists_graph = Z.evaluate_existential<1, 2>([](const torch::Tensor& cell) {
        return (cell > 1.5f) | (cell < -1.5f); // Maschera logica cellulare OR
    });

    STATIC_REQUIRE(decltype(exists_graph)::Rank == 2);
    STATIC_REQUIRE(decltype(exists_graph)::Shape[0] == 16);
    STATIC_REQUIRE(decltype(exists_graph)::Shape[1] == 3);
    std::cout << "-> [∃ OK] Tipo calcolato: MetaTensor<float, 16, 3> | VRAM: " 
              << exists_graph.storage.sizes() << "\n";

    // =========================================================================
    // TEST 2: Operatore di Aggregazione Relazionale Complementare (MapReduce)
    // =========================================================================
    // Ordiniamo al motore di TRATTENERE IMMUTATI solo l'asse 0 (16) e l'asse 2 (64).
    // Questo significa che l'asse 1 (32) e l'asse 3 (3) costituiscono il complemento 
    // e verranno contratti effettuando una riduzione moltiplicativa (PRODUCT).
    // Tipo atteso dedotto dal compilatore: MetaTensor<float, 16, 64> (Rank = 2)
    auto aggregated_graph = Z.aggregate<0, 2>(MetaTensor<float, 16, 32, 64, 3>::AggregationOp::PRODUCT);

    STATIC_REQUIRE(decltype(aggregated_graph)::Rank == 2);
    STATIC_REQUIRE(decltype(aggregated_graph)::Shape[0] == 16);
    STATIC_REQUIRE(decltype(aggregated_graph)::Shape[1] == 64);
    std::cout << "-> [AGGREGATE OK] Tipo calcolato: MetaTensor<float, 16, 64> | VRAM: " 
              << aggregated_graph.storage.sizes() << "\n";

    // Cleanup deterministico hardware Zero-Caching
    Z.clear();
    exists_graph.clear();
    aggregated_graph.clear();

}

TEST_CASE("Validazione Operatori Generalizzati e Valori", "[ex+agg]") {
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // 1. Inizializziamo un tensore compatto interamente a ZERO per avere il controllo totale dei dati
    // Dimensioni: [Batch=2, Rows=3, Cols=2, Channels=2] -> 24 elementi totali
    MetaTensor<float, 2, 3, 2, 2> Z( device, InitPattern::Zeros);

    // =========================================================================
    // INIEZIONE CHIRURGICA DEI DATI (Uso degli indici multidimensionali)
    // =========================================================================
    // Iniettiamo valori che attiveranno la maschera condizionale (cell > 1.5 o cell < -1.5)
    // Ricorda: l'esistenziale ridurrà l'asse 1 (Rows=3) e l'asse 2 (Cols=2)
    Z[std::array<size_t, 4>{0, 1, 0, 0}] = 2.0f;   // Attiva la cella per Batch=0, Channel=0
    Z[std::array<size_t, 4>{1, 2, 1, 1}] = -3.0f;  // Attiva la cella per Batch=1, Channel=1

    // Prepariamo anche una cella per il test di aggregazione (PRODUCT)
    // Impostiamo l'intera colonna/complemento per una determinata coordinata trattenuta
    // Vogliamo che l'aggregazione su (Batch=0, Cols=1) moltiplichi dei valori specifici
    // Gli assi da contrarre sono l'asse 1 (Rows) e l'asse 3 (Channels)
    Z[std::array<size_t, 4>{0, 0, 1, 0}] = 2.0f;
    Z[std::array<size_t, 4>{0, 1, 1, 0}] = 3.0f;
    Z[std::array<size_t, 4>{0, 2, 1, 0}] = 1.0f;
    // Tutte le altre celle non toccate rimangono a 0.0f grazie a InitPattern::Zeros

    // =========================================================================
    // TEST 1: Quantificatore Esistenziale Multi-Asse Generalizzato (∃)
    // =========================================================================
    // Riduciamo contemporaneamente l'asse 1 e l'asse 2.
    // Tipo atteso: MetaTensor<float, 2, 2> (ovvero Batch e Channels rimasti)
    auto exists_graph = Z.template evaluate_existential<1, 2>([](const torch::Tensor& cell) {
        return (cell > 1.5f) | (cell < -1.5f);
    });

    // Convalida formale dei metatipi
    STATIC_REQUIRE(decltype(exists_graph)::Rank == 2);
    STATIC_REQUIRE(decltype(exists_graph)::Shape[0] == 2);
    STATIC_REQUIRE(decltype(exists_graph)::Shape[1] == 2);

    // Convalida Matematica dei Valori a runtime
    auto cpu_exists = exists_graph.to_host();
    auto val1 = cpu_exists[std::array<size_t, 2>{0, 0}];
    auto val2 = cpu_exists[std::array<size_t, 2>{0, 1}];
    auto val3 = cpu_exists[std::array<size_t, 2>{1, 0}];
    auto val4 = cpu_exists[std::array<size_t, 2>{1, 1}];

    // Per Batch=0, Channel=0 -> Esiste l'elemento 2.0f (iniettato in {0,1,0,0}) -> Deve essere 1.0f (True)
    REQUIRE_THAT(val1, Catch::Matchers::WithinAbs(1.0f, 1e-5f));

    // Per Batch=0, Channel=1 -> Nessun elemento attivante inserito -> Deve essere 0.0f (False)
    REQUIRE_THAT(val2, Catch::Matchers::WithinAbs(0.0f, 1e-5f));

    // Per Batch=1, Channel=0 -> Nessun elemento attivante inserito -> Deve essere 0.0f (False)
    REQUIRE_THAT(val3, Catch::Matchers::WithinAbs(0.0f, 1e-5f));

    // Per Batch=1, Channel=1 -> Esiste l'elemento -3.0f (iniettato in {1,2,1,1}) -> Deve essere 1.0f (True)
    REQUIRE_THAT(val4, Catch::Matchers::WithinAbs(1.0f, 1e-5f));

    // =========================================================================
    // TEST 2: Operatore di Aggregazione Relazionale Complementare (PRODUCT)
    // =========================================================================
    // Tratteniamo l'asse 0 (Batch=2) e l'asse 2 (Cols=2).
    // Gli assi complemento contratti sono l'asse 1 (Rows=3) e l'asse 3 (Channels=2).
    // Tipo atteso: MetaTensor<float, 2, 2>
    auto aggregated_graph = Z.template aggregate<0, 2>(decltype(Z)::AggregationOp::PRODUCT);

    STATIC_REQUIRE(decltype(aggregated_graph)::Rank == 2);
    STATIC_REQUIRE(decltype(aggregated_graph)::Shape[0] == 2);
    STATIC_REQUIRE(decltype(aggregated_graph)::Shape[1] == 2);

    auto cpu_agg = aggregated_graph.to_host();

    // Ragionamento sul prodotto delle celle coordinate per {Batch=0, Col=1}:
    // Gli elementi lungo gli assi controllati sono:
    // {0,0,1,0}=2.0, {0,0,1,1}=0.0, {0,1,1,0}=3.0, {0,1,1,1}=0.0, {0,2,1,0}=1.0, {0,2,1,1}=0.0
    // Poiché ci sono degli zeri strutturali non toccati dall'inizializzazione,
    // il prodotto totale della contrazione di quella specifica fetta relazionale deve collassare a 0.0f
    REQUIRE_THAT((cpu_agg[std::array<size_t, 2>{0, 1}]), Catch::Matchers::WithinAbs(0.0f, 1e-5f));

    // Cambiamo approccio per verificare una riduzione moltiplicativa pulita priva di zeri:
    // Riempiamo un micro-tensore interamente a 2.0f per testare la riduzione geometrica pura
    MetaTensor<float, 2, 2, 2, 1> Ones_Tensor(device, InitPattern::OnOnes);
    auto Double_Tensor = Ones_Tensor * 2.0f; // Ogni cella fisica vale 2.0f

    // Tratteniamo solo l'asse 0 (Dim=2) e l'asse 3 (Dim=1). Gli assi contratti sono l'asse 1 (Dim=2) e l'asse 2 (Dim=2)
    // Numero di elementi contratti per ogni combinazione: 2 * 2 = 4 elementi.
    // Calcolo atteso: 2.0f ^ 4 = 16.0f
    auto prod_check = Double_Tensor.template aggregate<0, 3>(decltype(Double_Tensor)::AggregationOp::PRODUCT);
    auto cpu_prod_check = prod_check.to_host();

    REQUIRE_THAT((cpu_prod_check[std::array<size_t, 2>{0, 0}]), Catch::Matchers::WithinAbs(16.0f, 1e-5f));
    REQUIRE_THAT((cpu_prod_check[std::array<size_t, 2>{1, 0}]), Catch::Matchers::WithinAbs(16.0f, 1e-5f));

    // Cleanup deterministico hardware Zero-Caching
    Z.clear();
    Ones_Tensor.clear();
    Double_Tensor.clear();
    exists_graph.clear();
    aggregated_graph.clear();
    prod_check.clear();
}


#include <torch/torch.h>
#include <logds/metatensor/MetaTensor.h>
#include <logds/metatensor/GradientTape.h>
#include <logds/metatensor/CellOp.h>
#include <iostream>

TEST_CASE("Validazione Operatori Unari", "[unop]")  {
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "=== Verifica Operatore Unario Unificato MetaTensor ===\n\n";

    MetaTensor<float, 128, 64> X(device, InitPattern::RandomUniform);
    MetaTensor<float, 64, 1>   W( device, InitPattern::Zeros);

    // 1. FORWARD PASS PULITO ED UNIFICATO
    // Invece di chiamare metodi hardcoded, indichiamo l'operazione tramite l'enum
    auto linear_projection = X * W;
    
    // Calcoliamo la Sigmoide element-wise sul chip grafico
    auto Y_pred = linear_projection.apply<CellOp::Sigmoid>();
    
    // Se volessimo calcolare una funzione di attivazione alternativa (es: Tanh) nello stesso punto:
    auto Y_pred_tanh = linear_projection.apply<CellOp::Tanh>();

    std::cout << "-> [SUCCESS] Trasformazioni unarie applicate correttamente in VRAM.\n";
    std::cout << "   Forma dell'esito Sigmoide: " << Y_pred.storage.sizes() << "\n";
    std::cout << "   Forma dell'esito Tanh:     " << Y_pred_tanh.storage.sizes() << "\n";

    // Pulizia Zero-Caching
    X.clear();
    W.clear();
    Y_pred.clear();
    Y_pred_tanh.clear();
}

