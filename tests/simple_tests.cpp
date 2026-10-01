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
