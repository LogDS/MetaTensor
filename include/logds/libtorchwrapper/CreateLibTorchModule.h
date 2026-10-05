//
// Created by gyankos on 06/10/26.
//

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
