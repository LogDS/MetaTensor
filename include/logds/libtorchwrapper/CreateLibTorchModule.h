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
 * along with this program. If not, see <http://gnu.org>.
 */

#ifndef METATENSOR_CREATELIBTORCHMODULE_H
#define METATENSOR_CREATELIBTORCHMODULE_H

#include <string>
#include <torch/script.h>
#include <iostream>
#include "logds/libtorchwrapper/CreateMethod.h"


namespace libtorchwrapper {
    class CreateLibTorchModule {
    std::string name;
    torch::jit::Module module;
        std::unordered_map<std::string, CreateMethod> modules;

public:
    CreateLibTorchModule(const std::string& name);

    CreateMethod& createMethod(const std::string& name = "forward") {
        auto it = modules.find(name);
        if (it == modules.end()) {
            it = modules.emplace(name, CreateMethod{&module, name}).first;
        }
        return it->second;
    }

    void compile(const std::string& filename) const { module.save(filename); }
};
} // libtorchwrapper

#endif //METATENSOR_CREATELIBTORCHMODULE_H
