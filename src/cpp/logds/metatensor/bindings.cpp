/*
 * This file is part of the MetaTensor distribution (https://github.com).
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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/torch.h>
#include <logds/metatensor/MetaTensor.h>
#include <logds/metatensor/Init.h>
#include <string>
#include <sstream>
#include <array>
#include <tuple>

namespace nb = nanobind;

// Helper strutturali per definire le liste di dimensioni fisse a compile-time
template <size_t... Is> struct ShapePack {
    static constexpr std::array<size_t, sizeof...(Is)> data = { Is... };
};

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

    auto cl = nb::class_<TensorType>(m, name.c_str())
        // Costruttore standard che accetta il tag InitPattern e il dispositivo
        .def(nb::init<torch::Device, InitPattern>(),  nb::arg("device") = torch::kCPU, nb::arg("pattern") = InitPattern::RandomNormal)
        .def_ro_static("rank", &TensorType::Rank)
        .def_prop_ro("shape", [](const TensorType&) {
            std::vector<size_t> s(TensorType::Shape.begin(), TensorType::Shape.end());
            return s;
        })
        .def("watch", &TensorType::watch)
        .def("clear", &TensorType::clear)
        .def("element_wise_sigmoid", &TensorType::element_wise_sigmoid)
        .def("reduce_all_sum", &TensorType::reduce_all_sum)
        .def("pretty_print_sparse", &TensorType::pretty_print_sparse, nb::arg("threshold") = 1e-4f)


        // Interfaccia esplicita per l'applicazione dei gradienti
        .def("apply_gradient_descent", [](TensorType& t, const TensorType& grad, float lr) {
            t.apply_gradient_descent(grad, lr);
        });

    // RISOLUTIVO: Il compilatore inietta __float__ in Python SOLO se la variante
    // ha Rank == 0 o se tutte le sue dimensioni statiche collassano a 1.
    if constexpr (TensorType::Rank == 0 || (... && (Dims == 1))) {
        cl.def("__float__", [](const TensorType& t) {
            // Invocazione sicura del cast implicito convalidato a compile-time
            return static_cast<float>(t);
        });
    }
}

// 3. Helper di scompattamento: estrae in modo pulito il pack "Is..." da ShapePack
template <typename T, typename SingleShape>
struct SingleShapeBinder;

template <typename T, size_t... Is>
struct SingleShapeBinder<T, ShapePack<Is...>> {
    static void execute(nb::module_ &m) {
        bind_tensor_instance<T, Is...>(m);
    }
};

// 4. Meta-Binder Loop per srotolare la tupla di ShapePack
template <typename T, typename ShapesTuple>
struct MetaTensorBinder;

template <typename T, typename... SubShapes>
struct MetaTensorBinder<T, std::tuple<SubShapes...>> {
    static void bind(nb::module_ &m) {
        // Il fold expression ora agisce sulla classe helper, dove "SubShapes" è un pack di tipi valido
        (SingleShapeBinder<T, SubShapes>::execute(m), ...);
    }
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
