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

#include <logds/metatensor/MetaTensor.h>


// =============================================================================
// VERIFICA SINTATTICA IN C++26
// =============================================================================
int main() {
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    std::cout << "=== Inizializzazione Wrapper MetaTensor (Eigen Style) ===\n";

    // Allocazione tensori fortemente tipizzati con dimensioni fisse nel tipo
    DMetaTensor<float, 16, 32, 64> t1(device); // Left Tensor  [Rank 3]
    DMetaTensor<float, 128, 64>    t2(device); // Right Tensor [Rank 2]

    std::cout << "-> Tensore t1 [Left] instanziato con rango: " << t1.Rank << "\n";
    std::cout << "-> Tensore t2 [Right] instanziato con rango: " << t2.Rank << "\n";

    // Eseguiamo la contrazione relazionale proiettando solo gli assi desiderati
    // Selezioniamo: asse 0 di L (16), asse 1 di L (32), e asse 0 di R (128)
    // Il compilatore deduce che l'asse 2 di L (64) e l'asse 1 di R (64) si contraggono e valida la build.
    auto t_result = t1.contraction<L<0>, L<1>, R<0>>(t2);

    std::cout << "\n=== Compilazione ed Esecuzione OK ===" << "\n";
    std::cout << "Firma del Tipo risultante dedotta a compile-time:\n";
    std::cout << "-> MetaTensor<float";
    for(size_t s : t_result.Shape) std::cout << ", " << s;
    std::cout << ">\n";

    std::cout << "Forma fisica effettiva in VRAM: " << t_result.storage.sizes() << "\n";

    // =========================================================================
    // DIMOSTRAZIONE DELLA PROTEZIONE DI COMPILAZIONE (STATIC_ASSERT)
    // =========================================================================
    // Rimuovi il commento dalla riga sottostante per testare la protezione:
    // Tenta di contrarre un asse da 32 con uno da 64 -> Genera un Errore di Compilazione bloccante!
    // MetaTensor<float, 16, 64, 32> t_error(device);
    // auto t_fail = t_error.contraction<L<0>, L<1>, R<0>>(t2);

    return 0;
}
