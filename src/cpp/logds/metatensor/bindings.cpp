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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/torch.h>
#include <logds/metatensor/MetaTensor.h>
#include <logds/metatensor/Init.h>
#include <string>
#include <sstream>

namespace nb = nanobind;

// 1. Generatore Constexpr del nome della classe Python (es: "MetaTensor_float_128_64")
template <typename T, size_t... Dims>
std::string get_tensor_classname() {
    std::stringstream ss;
    if constexpr (std::is_same_v<T, float>) ss << "MetaTensor_float";
    else if constexpr (std::is_same_v<T, double>) ss << "MetaTensor_double";

    ((ss << "_" << Dims), ...);
    return ss.str();
}

// 2. Registro a basso livello della singola variante strutturata
template <typename T, size_t... Dims>
void bind_tensor_instance(nb::module_ &m) {
    std::string name = get_tensor_classname<T, Dims...>();
    using TensorType = MetaTensor<T, Dims...>;

    nb::class_<TensorType>(m, name.c_str())
        // Costruttore standard che accetta il tag InitPattern e il dispositivo
        .def(nb::init<InitPattern, torch::Device>(), nb::arg("pattern"), nb::arg("device") = torch::kCPU)
        // Costruttore di default (RandomNormal)
        .def(nb::init<torch::Device>(), nb::arg("device") = torch::kCPU)

        .def_readonly_static("rank", &TensorType::Rank)
        .def_property_readonly("shape", [](const TensorType&) {
            std::vector<size_t> s(TensorType::Shape.begin(), TensorType::Shape.end());
            return s;
        })

        .def("watch", &TensorType::watch)
        .def("clear", &TensorType::clear)
        .def("element_wise_sigmoid", &TensorType::element_wise_sigmoid)
        .def("reduce_all_sum", &TensorType::reduce_all_sum)
        .def("pretty_print_sparse", &TensorType::pretty_print_sparse, nb::arg("threshold") = 1e-4f)

        // Estrattore scalare nativo per Python via cast implicito C++26
        .def("__float__", [](const TensorType& t) { return static_cast<float>(t); })

        // Interfaccia esplicita per l'applicazione dei gradienti
        .def("apply_gradient_descent", [](TensorType& t, const TensorType& grad, float lr) {
            t.apply_gradient_descent(grad, lr);
        });
}

// 3. Meta-Binder Loop per srotolare coppie e combinazioni di dimensioni esplicite
template <typename T, typename ShapesTuple>
struct MetaTensorBinder;

template <typename T, typename... SubShapes>
struct MetaTensorBinder<T, std::tuple<SubShapes...>> {
    static void bind(nb::module_ &m) {
        // Srotola le definizioni della tupla invocando il binding istanziato
        ([&]() {
            bind_tensor_instance<T, SubShapes::data...>(m);
        }(), ...);
    }
};

// Helper strutturali per definire le liste di dimensioni fisse a compile-time
template <size_t... Is> struct ShapePack {
    static constexpr std::array<size_t, sizeof...(Is)> data = { Is... };
};

// =============================================================================
// MODULO CORE DEFINITIVO DI NANOBIND
// =============================================================================
NB_MODULE(metatensor_core, m) {
    // Esponiamo l'enumeratore degli InitPattern in Python per preservare la semantica
    nb::enum_<InitPattern>(m, "InitPattern")
        .value("RandomNormal", InitPattern::RandomNormal)
        .value("RandomUniform", InitPattern::RandomUniform)
        .value("Zeros", InitPattern::Zeros)
        .value("OnOnes", InitPattern::OnOnes)
        .value("Identity", InitPattern::Identity);

    // Definiamo RIGIDAMENTE la griglia geometrica delle varianti ammesse nel binario
    // Questo mappa al 100% tutte le combinazioni richieste dal tuo ciclo di addestramento
    using TargetShapes = std::tuple<
        ShapePack<128, 64>, // Dataset di Input X
        ShapePack<128, 1>,  // Target reali Y_true e bias b
        ShapePack<64, 1>,   // Pesi del modello W e gradiente dW
        ShapePack<64, 64>,  // Matrice Identità I
        ShapePack<>         // Loss scalare 0-D (Rank = 0)
    >;

    // Compiliamo ed iniettiamo le classi nel modulo binario
    MetaTensorBinder<float, TargetShapes>::bind(m);
}

