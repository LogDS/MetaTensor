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

#ifndef TENSORLIBRARY_DISTRIBUTEDCONTEXT_H
#define TENSORLIBRARY_DISTRIBUTEDCONTEXT_H

#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <string>
#include <vector>

// RISOLUTIVO: Header core C++ di basso livello sempre garantiti presenti in LibTorch
#ifdef METATENSOR_USE_MPI
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#endif

enum class DistMode { SingleMachine, MultiNodeMPI };

class DistributedContext {
private:
    inline static DistMode mode = DistMode::SingleMachine;
    inline static int world_size = 1;
    inline static int rank = 0;
    inline static bool initialized = false;

#ifdef METATENSOR_USE_MPI
    // Manteniamo il riferimento al gruppo di processo core C++ condiviso
    inline static c10::intrusive_ptr<c10d::ProcessGroupGloo> process_group;
#endif

public:
    static void init() {
        if (initialized) return;

#ifdef METATENSOR_USE_MPI
        // Rilevamento runtime del lancio tramite gestore processi (mpirun/mpiexec)
        const char* mpi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
        const char* mpi_size = std::getenv("OMPI_COMM_WORLD_SIZE");
        if (!mpi_rank) mpi_rank = std::getenv("PMI_RANK");
        if (!mpi_size) mpi_size = std::getenv("PMI_SIZE");

        if (mpi_rank && mpi_size) {
            mode = DistMode::MultiNodeMPI;
            rank = std::atoi(mpi_rank);
            world_size = std::atoi(mpi_size);

            std::string init_method = "/tmp/metatensor_shared_sync_store";
            if (rank == 0) { std::remove(init_method.c_str()); }

            // 1. Creazione dello Store di coordinamento basato su file condiviso
            auto store = c10::make_intrusive<c10d::FileStore>(init_method, world_size);

            // 2. RISOLUTIVO: Istanziazione diretta del ProcessGroup C++ nativo (Gloo Core)
            // Questo bypassa il namespace astratto torch::distributed, eliminando l'errore del compilatore.
            process_group = c10::make_intrusive<c10d::ProcessGroupGloo>(store, rank, world_size);

            std::cout << "[METATENSOR CLUSTER] Node Rank " << rank << " / " << world_size
                      << " initialized successfully via Gloo C++ Core backend.\n";
            initialized = true;
            return;
        }
#endif

        // FALLBACK DI DEFAULT: Eseguito se siamo su macchina singola locale
        mode = DistMode::SingleMachine;
        rank = 0;
        world_size = 1;
        std::cout << "[METATENSOR] Running in Single-Machine mode. Fallback to local hardware.\n";
        initialized = true;
    }

    static DistMode get_mode() { return mode; }
    static int get_rank() { return rank; }
    static int get_world_size() { return world_size; }
    static bool is_root() { return rank == 0; }
    static bool is_distributed() { return mode == DistMode::MultiNodeMPI; }

#ifdef METATENSOR_USE_MPI
    static c10::intrusive_ptr<c10d::ProcessGroupGloo> get_group() { return process_group; }
#endif
};

#endif //TENSORLIBRARY_DISTRIBUTEDCONTEXT_H
