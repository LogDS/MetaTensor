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

int main() {
    // 1. Verifichiamo se l'hardware CUDA è disponibile nel sistema
    torch::Device gpu_device(torch::kCUDA);
    torch::Device cpu_device(torch::kCPU);

    if (!torch::cuda::is_available()) {
        std::cerr << "[WARNING] CUDA non rilevato. Il test eseguirà i trasferimenti simulati.\n";
        gpu_device = torch::kCPU;
    }

    std::cout << "=== Test del Trasferimento Memoria Host-Device (C++26) ===\n\n";

    // 2. Inizializzazione: Il tensore nasce sulla CPU (Host Memory)
    std::cout << "-> Inizializzazione Tensore su CPU...\n";
    MetaTensor<float, 2, 4, 3> tensor_host(cpu_device);
    std::cout << "   Ubicazione iniziale: " << tensor_host.storage.device() << "\n\n";

    // 3. OFFLOADING VERSO LA GPU: Spingiamo il tensore in VRAM per la computazione
    std::cout << "-> Esecuzione .to_device() -> Spostamento dei vettori in VRAM...\n";
    auto tensor_gpu = tensor_host.to_device(gpu_device);
    std::cout << "   Nuova ubicazione hardware: " << tensor_gpu.storage.device() << "\n\n";

    // [Qui l'orchestratore host lancia i macro-kernel, matmul, o theta-joins sulla GPU]
    auto calculated_gpu = tensor_gpu.element_wise_sigmoid();

    // 4. RETRIEVAL VERSO LA CPU: Scarichiamo l'esito finale per il Dump Sparso
    std::cout << "-> Esecuzione .to_host() -> Recupero dati ed evacuazione VRAM...\n";
    auto final_result_host = calculated_gpu.to_host();
    std::cout << "   Ubicazione finale dei dati: " << final_result_host.storage.device() << "\n";

    // Il dump viene eseguito in modo sicuro sulla memoria centrale (RAM)
    final_result_host.pretty_print_sparse(0.1f);

    // =========================================================================
    // TRASPARENZA DELLO SCOPE E ANTI-CACHING
    // =========================================================================
    // Alla chiusura del programma o dello scope {}, i distruttori distruggono i riferimenti.
    // Grazie alla chiamata a emptyCache() integrata nel flusso di .to_host() e dei distruttori,
    // la scheda video viene completamente liberata da ogni blocco latente in modo sincrono.

    return 0;
}
