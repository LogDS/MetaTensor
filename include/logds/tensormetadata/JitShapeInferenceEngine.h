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

#ifndef METATENSOR_JITSHAPEINFERENCEENGINE_H
#define METATENSOR_JITSHAPEINFERENCEENGINE_H

#include <unordered_map>
#include <logds/tensormetadata/Metadata.h>


class JitShapeInferenceEngine {
private:
    // Registro Union-Find per l'unificazione e la risoluzione dei simboli a runtime
    mutable std::unordered_map<std::string, std::string> symbol_aliases;
    mutable std::unordered_map<std::string, uint64_t> symbolic_numeric_bounds;

    // Trova il rappresentante canonico di un simbolo (Path Compression)
    std::string find_symbol_root(const std::string &sym) const;

    // RISOLUTIVO: Unifica due dimensioni o ne verifica la conformità contestuale
    bool dims_match(const std::variant<std::string, uint64_t> &d1,
                    const std::variant<std::string, uint64_t> &d2) const;

    bool is_one(const std::variant<std::string, uint64_t> &d) const;

    JitTensorMetadata make_scalar(DataType dt) const {
        return JitTensorMetadata{dt, StorageLayout::Dense, 0, {}, true};
    }

public:
    JitShapeInferenceEngine() = default;

    // Ritorna il simbolo normalizzato o il numero risolto per l'esportazione pulita del grafo
    std::variant<std::string, uint64_t>
    resolve_canonical_dimension(const std::variant<std::string, uint64_t> &d) const;

    // 1. PERMUTE_AXES
    std::optional<JitTensorMetadata> permute_axes(const JitTensorMetadata &in, const std::vector<size_t> &perm) const;

    // 2. REDUCE_SUM
    std::optional<JitTensorMetadata> reduce_sum(const JitTensorMetadata &in, size_t axis) const;

    // 3. REDUCE_ALL_SUM
    std::optional<JitTensorMetadata> reduce_all_sum(const JitTensorMetadata &in) const;

    // 4. CASTING
    std::optional<JitTensorMetadata> casting(const JitTensorMetadata &in, DataType target_type) const;

    // 5. FLOAT EXTRACTION
    std::optional<JitTensorMetadata> float_extraction(const JitTensorMetadata &in) const;

    // 6. SLICE_TENSOR
    std::optional<JitTensorMetadata> slice_tensor(const JitTensorMetadata &in, size_t axis, uint64_t start,
                                                  uint64_t end) const;

    // 7. SQUEEZE_AXES
    std::optional<JitTensorMetadata> squeeze_axes(const JitTensorMetadata &in, size_t axis) const;

    // 8. SEGMENT_SUM (Sorgente 2D [Rows, Cols] -> Destinazione 2D [NumSegments, Cols])
    std::optional<JitTensorMetadata> segment_sum(const JitTensorMetadata& in, uint64_t num_segments) const;

    // 9. ONE_HOT_ENCODING (Vettore 1D [Len] -> Matrice 2D [Len, Depth])
    std::optional<JitTensorMetadata> one_hot_encoding(const JitTensorMetadata& in, uint64_t depth) const;

    // 14. OPERATOR* (Contrazione d'asse MatMul: Left [M, K] * Right [K, N] -> Output [M, N])
    std::optional<JitTensorMetadata> operator_matmul(const JitTensorMetadata& left, const JitTensorMetadata& right) const;

    // 15. OPERATOR% (Cross Product: Left [3] % Right [3] -> Output)
    std::optional<JitTensorMetadata> operator_cross(const JitTensorMetadata& left, const JitTensorMetadata& right) const;


    // // 8. SEGMENT_SUM
    // std::optional<JitTensorMetadata> segment_sum(const JitTensorMetadata &in, uint64_t num_segments) const {
    //     if (!in.is_valid || in.rank != 2) return std::nullopt;
    //     return JitTensorMetadata{in.data_type, in.layout, 2, std::vector<std::variant<std::string, uint64_t> >{num_segments, in.shape}, true};
    // }
    //
    // // 9. ONE_HOT_ENCODING
    // std::optional<JitTensorMetadata> one_hot_encoding(const JitTensorMetadata &in, uint64_t depth) const {
    //     if (!in.is_valid || in.rank != 1) return std::nullopt;
    //     return JitTensorMetadata{TYPE_FLOAT, StorageLayout::Dense, 2, {in.shape, depth}, true};
    // }

    // 10. GATHER_ND
    std::optional<JitTensorMetadata>
    gather_nd(const JitTensorMetadata &source, const JitTensorMetadata &indices) const;

    // 11. TENSOR_THETA_JOIN
    // 11. TENSOR_THETA_JOIN (Left [LenA] x Right [LenB] -> Output Matrix 2D [WorstCaseMaxPairs, 2])
    std::optional<JitTensorMetadata> tensor_theta_join(const JitTensorMetadata& left, const JitTensorMetadata& right) const;


    // 12. EVALUATE_EXISTENTIAL_EXPRESSION
    std::optional<JitTensorMetadata> evaluate_existential_expression(const JitTensorMetadata &in, size_t axis) const {
        auto out = reduce_sum(in, axis);
        if (out) out->data_type = TYPE_FLOAT;
        return out;
    }

    // 13. OPERATOR+ (Unificazione e Broadcasting Simbolico)
    std::optional<JitTensorMetadata> operator_add(const JitTensorMetadata &left, const JitTensorMetadata &right) const;

    // // 14. OPERATOR* (Contrazione d'asse MatMul con Unificazione)
    // std::optional<JitTensorMetadata> operator_matmul(const JitTensorMetadata &left,
    //                                                  const JitTensorMetadata &right) const {
    //     if (!left.is_valid || !right.is_valid || left.data_type != right.data_type || left.rank < 2 || right.rank < 2)
    //         return std::nullopt;
    //     // RISOLUTIVO: Vincola e unifica l'asse interno di contrazione (es: InDim == InDim o Batch == M)
    //     if (!dims_match(left.shape[left.rank - 1], right.shape)) return std::nullopt;
    //     return JitTensorMetadata{
    //         left.data_type, StorageLayout::Dense, 2,
    //         {resolve_canonical_dimension(left.shape), resolve_canonical_dimension(right.shape)}, true
    //     };
    // }
    //
    // // 15. OPERATOR%
    // std::optional<JitTensorMetadata>
    // operator_cross(const JitTensorMetadata &left, const JitTensorMetadata &right) const {
    //     if (!left.is_valid || !right.is_valid || left.rank != 1 || right.rank != 1) return std::nullopt;
    //     if (!dims_match(left.shape, uint64_t(3)) || !dims_match(right.shape, uint64_t(3))) return std::nullopt;
    //     return JitTensorMetadata{left.data_type, StorageLayout::Dense, 1, {uint64_t(3)}, true};
    // }

    // 16. CONTRACTION (Einsum Relazionale Astratta con Unificazione)
    std::optional<JitTensorMetadata> contraction(const JitTensorMetadata &left, const JitTensorMetadata &right,
                                                 size_t left_axis, size_t right_axis) const;

    // 17. WHERE
    std::optional<JitTensorMetadata> where(const JitTensorMetadata &in, float off_value) const;

    // 10. OPERATOR-
    std::optional<JitTensorMetadata> operator_sub(const JitTensorMetadata &left, const JitTensorMetadata &right) const;

    // 19. ELEMENT_WISE_MUL
    std::optional<JitTensorMetadata> element_wise_mul(const JitTensorMetadata &left,
                                                      const JitTensorMetadata &right) const;

    // 20. OPERATOR[]
    std::optional<JitTensorMetadata> operator_brackets(const JitTensorMetadata &in,
                                                       const std::vector<uint64_t> &coords) const;

    // 21. APPLY
    std::optional<JitTensorMetadata> apply(const JitTensorMetadata &in, const std::string &op_name) const;

    // 22. EVALUATE_EXISTENTIAL
    std::optional<JitTensorMetadata> evaluate_existential(const JitTensorMetadata &in,
                                                          const std::vector<size_t> &reduce_axes) const;

    // 23. EVALUATE_UNIVERSAL
    std::optional<JitTensorMetadata> evaluate_universal(const JitTensorMetadata &in,
                                                        const std::vector<size_t> &reduce_axes) const;

    // 24. AGGREGATE
    std::optional<JitTensorMetadata> aggregate(const JitTensorMetadata &in,
                                               const std::vector<size_t> &retained_axes) const;
};

//
// class JitShapeInferenceEngine {
// private:
//     // Helper per verificare se due assi simbolici/numerici coincidono matematicamente
//     static bool dims_match(const std::variant<std::string, uint64_t> &d1,
//                            const std::variant<std::string, uint64_t> &d2) {
//         if (d1.index() != d2.index()) return false;
//         if (std::holds_alternative<uint64_t>(d1)) {
//             return std::get<uint64_t>(d1) == std::get<uint64_t>(d2);
//         }
//         return std::get<std::string>(d1) == std::get<std::string>(d2);
//     }
//
//     // Helper per verificare se un asse equivale alla costante numerica 1 (Broadcasting)
//     static bool is_one(const std::variant<std::string, uint64_t> &d) {
//         return std::holds_alternative<uint64_t>(d) && std::get<uint64_t>(d) == 1;
//     }
//
//     // Helper per creare un'istanza pulita di uno scalare puro 0-D (Loss/Casts)
//     static JitTensorMetadata make_scalar(DataType dt) {
//         return JitTensorMetadata{dt, StorageLayout::Dense, 0, {}, true};
//     }
//
// public:
//     // 1. PERMUTE_AXES
//     static std::optional<JitTensorMetadata> permute_axes(const JitTensorMetadata &in, const std::vector<size_t> &perm) {
//         if (!in.is_valid || perm.size() != in.rank) return std::nullopt;
//
//         std::vector<bool> checked(in.rank, false);
//         std::vector<std::variant<std::string, uint64_t> > out_shape(in.rank);
//
//         for (size_t i = 0; i < perm.size(); ++i) {
//             size_t axis = perm[i];
//             if (axis >= in.rank || checked[axis]) return std::nullopt;
//             checked[axis] = true;
//             out_shape[i] = in.shape[axis];
//         }
//         return JitTensorMetadata{in.data_type, in.layout, in.rank, out_shape, true};
//     }
//
//     // 2. REDUCE_SUM
//     static std::optional<JitTensorMetadata> reduce_sum(const JitTensorMetadata &in, size_t axis) {
//         if (!in.is_valid || in.rank == 0 || axis >= in.rank) return std::nullopt;
//
//         std::vector<std::variant<std::string, uint64_t> > out_shape;
//         for (size_t i = 0; i < in.rank; ++i) {
//             if (i != axis) out_shape.push_back(in.shape[i]);
//         }
//         return JitTensorMetadata{in.data_type, in.layout, in.rank - 1, out_shape, true};
//     }
//
//     // 3. REDUCE_ALL_SUM
//     static std::optional<JitTensorMetadata> reduce_all_sum(const JitTensorMetadata &in) {
//         if (!in.is_valid) return std::nullopt;
//         return make_scalar(in.data_type); // Collassa a scalare puro 0-D
//     }
//
//     // 4. CASTING
//     static std::optional<JitTensorMetadata> casting(const JitTensorMetadata &in, DataType target_type) {
//         if (!in.is_valid) return std::nullopt;
//         return JitTensorMetadata{target_type, in.layout, in.rank, in.shape, true};
//     }
//
//     // 5. FLOAT EXTRACTION (Implicit primitive cast validation)
//     static std::optional<JitTensorMetadata> float_extraction(const JitTensorMetadata &in) {
//         if (!in.is_valid) return std::nullopt;
//         // Ammissibile solo se Rank == 0 o se tutte le dimensioni numeriche isolate sono pari a 1
//         bool convertible = (in.rank == 0);
//         if (in.rank > 0) {
//             convertible = std::all_of(in.shape.begin(), in.shape.end(), [](const auto &d) { return is_one(d); });
//         }
//         if (!convertible) return std::nullopt;
//         return make_scalar(TYPE_FLOAT);
//     }
//
//     // 6. SLICE_TENSOR
//     static std::optional<JitTensorMetadata> slice_tensor(const JitTensorMetadata &in, size_t axis, uint64_t start,
//                                                          uint64_t end) {
//         if (!in.is_valid || axis >= in.rank || end <= start) return std::nullopt;
//         if (std::holds_alternative<uint64_t>(in.shape[axis])) {
//             if (end > std::get<uint64_t>(in.shape[axis])) return std::nullopt;
//         }
//
//         std::vector<std::variant<std::string, uint64_t> > out_shape = in.shape;
//         out_shape[axis] = end - start;
//         return JitTensorMetadata{in.data_type, in.layout, in.rank, out_shape, true};
//     }
//
//     // 7. SQUEEZE_AXES
//     static std::optional<JitTensorMetadata> squeeze_axes(const JitTensorMetadata &in, size_t axis) {
//         if (!in.is_valid || axis >= in.rank || !is_one(in.shape[axis])) return std::nullopt;
//
//         std::vector<std::variant<std::string, uint64_t> > out_shape;
//         for (size_t i = 0; i < in.rank; ++i) {
//             if (i != axis) out_shape.push_back(in.shape[i]);
//         }
//         return JitTensorMetadata{in.data_type, in.layout, in.rank - 1, out_shape, true};
//     }
//
//     // 8. SEGMENT_SUM
//     static std::optional<JitTensorMetadata> segment_sum(const JitTensorMetadata &in, uint64_t num_segments) {
//         if (!in.is_valid || in.rank != 2) return std::nullopt; // Richiede sorgente 2D
//         return JitTensorMetadata{in.data_type, in.layout, 2, {num_segments, in.shape[1]}, true};
//     }
//
//     // 9. ONE_HOT_ENCODING
//     static std::optional<JitTensorMetadata> one_hot_encoding(const JitTensorMetadata &in, uint64_t depth) {
//         if (!in.is_valid || in.rank != 1) return std::nullopt; // Richiede etichette 1D
//         return JitTensorMetadata{TYPE_FLOAT, StorageLayout::Dense, 2, {in.shape[0], depth}, true};
//     }
//
//     // 10. GATHER_ND
//     static std::optional<JitTensorMetadata>
//     gather_nd(const JitTensorMetadata &source, const JitTensorMetadata &indices) {
//         if (!source.is_valid || !indices.is_valid || source.rank != 2 || indices.rank != 2) return std::nullopt;
//         return JitTensorMetadata{source.data_type, StorageLayout::Dense, 1, {indices.shape[0]}, true};
//     }
//
//     // 11. TENSOR_THETA_JOIN
//     static std::optional<JitTensorMetadata> tensor_theta_join(const JitTensorMetadata &left,
//                                                               const JitTensorMetadata &right) {
//         if (!left.is_valid || !right.is_valid || left.rank != 1 || right.rank != 1) return std::nullopt;
//
//         std::variant<std::string, uint64_t> worst_case_len;
//         if (std::holds_alternative<uint64_t>(left.shape[0]) && std::holds_alternative<uint64_t>(right.shape[0])) {
//             worst_case_len = std::get<uint64_t>(left.shape[0]) * std::get<uint64_t>(right.shape[0]);
//         } else {
//             std::string l_sym = std::holds_alternative<std::string>(left.shape[0])
//                                     ? std::get<std::string>(left.shape[0])
//                                     : std::to_string(std::get<uint64_t>(left.shape[0]));
//             std::string r_sym = std::holds_alternative<std::string>(right.shape[0])
//                                     ? std::get<std::string>(right.shape[0])
//                                     : std::to_string(std::get<uint64_t>(right.shape[0]));
//             worst_case_len = l_sym + "_x_" + r_sym;
//         }
//         return JitTensorMetadata{TYPE_FLOAT, StorageLayout::Dense, 2, {worst_case_len, uint64_t(2)}, true};
//     }
//
//     // 12. EVALUATE_EXISTENTIAL_EXPRESSION
//     static std::optional<JitTensorMetadata> evaluate_existential_expression(const JitTensorMetadata &in, size_t axis) {
//         auto out = reduce_sum(in, axis);
//         if (out) out->data_type = TYPE_FLOAT; // Converte a Float numerico
//         return out;
//     }
//
//     // 13. OPERATOR+ (Auto-Broadcasting dei Simboli)
//     static std::optional<JitTensorMetadata>
//     operator_add(const JitTensorMetadata &left, const JitTensorMetadata &right) {
//         if (!left.is_valid || !right.is_valid || left.data_type != right.data_type) return std::nullopt;
//         if (left.rank == 0) return right;
//         if (right.rank == 0) return left;
//
//         size_t max_rank = std::max(left.rank, right.rank);
//         std::vector<std::variant<std::string, uint64_t> > out_shape(max_rank);
//
//         for (size_t i = 0; i < max_rank; ++i) {
//             size_t l_idx = left.rank - 1 - i;
//             size_t r_idx = right.rank - 1 - i;
//             auto l_dim = (i < left.rank) ? left.shape[l_idx] : std::variant<std::string, uint64_t>(uint64_t(1));
//             auto r_dim = (i < right.rank) ? right.shape[r_idx] : std::variant<std::string, uint64_t>(uint64_t(1));
//
//             if (dims_match(l_dim, r_dim)) { out_shape[max_rank - 1 - i] = l_dim; } else if (
//                 is_one(l_dim)) { out_shape[max_rank - 1 - i] = r_dim; } else if (is_one(r_dim)) {
//                 out_shape[max_rank - 1 - i] = l_dim;
//             } else return std::nullopt; // Scontro algebrico simbolico irrisolvibile
//         }
//
//         StorageLayout out_layout = (left.layout == StorageLayout::SparseCOO && right.layout == StorageLayout::SparseCOO)
//                                        ? StorageLayout::SparseCOO
//                                        : StorageLayout::Dense;
//         return JitTensorMetadata{left.data_type, out_layout, max_rank, out_shape, true};
//     }
//
//     // 14. OPERATOR* (Contrazione d'asse MatMul polimorfa)
//     static std::optional<JitTensorMetadata> operator_matmul(const JitTensorMetadata &left,
//                                                             const JitTensorMetadata &right) {
//         if (!left.is_valid || !right.is_valid || left.data_type != right.data_type || left.rank < 2 || right.rank < 2)
//             return std::nullopt;
//         if (!dims_match(left.shape[left.rank - 1], right.shape[0])) return std::nullopt;
//
//         return JitTensorMetadata{left.data_type, StorageLayout::Dense, 2, {left.shape[0], right.shape[1]}, true};
//     }
//
//     // 15. OPERATOR% (Cross Product Spaziale)
//     static std::optional<JitTensorMetadata> operator_cross(const JitTensorMetadata &left,
//                                                            const JitTensorMetadata &right) {
//         if (!left.is_valid || !right.is_valid || left.rank != 1 || right.rank != 1) return std::nullopt;
//         if (!dims_match(left.shape[0], uint64_t(3)) || !dims_match(right.shape[0], uint64_t(3))) return std::nullopt;
//         return JitTensorMetadata{left.data_type, StorageLayout::Dense, 1, {uint64_t(3)}, true};
//     }
//
//     // 16. CONTRACTION (Einsum Relazionale Astratta)
//     static std::optional<JitTensorMetadata> contraction(const JitTensorMetadata &left, const JitTensorMetadata &right,
//                                      size_t left_axis, size_t right_axis) {
//         if (!left.is_valid || !right.is_valid || left_axis >= left.rank || right_axis >= right.rank) return
//                 std::nullopt;
//         if (!dims_match(left.shape[left_axis], right.shape[right_axis])) return std::nullopt;
//         std::vector<std::variant<std::string, uint64_t> > out_shape;
//         for (size_t i = 0; i < left.rank; ++i) if (i != left_axis) out_shape.push_back(left.shape[i]);
//         for (size_t i = 0; i < right.rank; ++i) if (i != right_axis) out_shape.push_back(right.shape[i]);
//         return JitTensorMetadata{left.data_type, StorageLayout::Dense, out_shape.size(), out_shape, true};
//     }
//
//     // 17. WHERE
//     static std::optional<JitTensorMetadata> where(const JitTensorMetadata &in, float off_value) {
//         if (!in.is_valid) return std::nullopt;
//         StorageLayout out_layout = (off_value == 0.0f) ? in.layout : StorageLayout::Dense;
//         return JitTensorMetadata{in.data_type, out_layout, in.rank, in.shape, true};
//     }
//
//     // 18. OPERATOR-
//     static std::optional<JitTensorMetadata> operator_sub(const JitTensorMetadata &left, const JitTensorMetadata &right) {
//         return operator_add(left, right);
//     }
//
//     // 19. ELEMENT_WISE_MUL
//     static std::optional<JitTensorMetadata> element_wise_mul(const JitTensorMetadata &left, const JitTensorMetadata &right) {
//         auto out = operator_add(left, right);
//         if (!out) return std::nullopt;
//         if (left.layout == StorageLayout::SparseCOO || right.layout == StorageLayout::SparseCOO) {
//             out->layout = StorageLayout::SparseCOO; // Intersezione preserva la parsimonia
//         }
//         return out;
//     }
//
//     // 20. OPERATOR[] (Cell Extraction Barrier)
//     static std::optional<JitTensorMetadata> operator_brackets(const JitTensorMetadata &in, const std::vector<uint64_t> &coords) {
//         if (!in.is_valid || coords.size() != in.rank) return std::nullopt;
//         for (size_t i = 0; i < in.rank; ++i) {
//             if (std::holds_alternative<uint64_t>(in.shape[i])) {
//                 if (coords[i] >= std::get<uint64_t>(in.shape[i])) return std::nullopt; // Out of bounds
//             }
//         }
//         return make_scalar(in.data_type);
//     }
//
//     // 21. APPLY (Unary Operator Layout Dispatching)
//     static std::optional<JitTensorMetadata> apply(const JitTensorMetadata &in, const std::string &op_name) {
//         if (!in.is_valid) return std::nullopt;
//         bool preserves_zero = (op_name == "Abs" || op_name == "Sqrt" || op_name == "Square" || op_name == "Tanh");
//         StorageLayout out_layout = preserves_zero ? in.layout : StorageLayout::Dense;
//         return JitTensorMetadata{in.data_type, out_layout, in.rank, in.shape, true};
//     }
//
//     // 22. EVALUATE_EXISTENTIAL
//     static std::optional<JitTensorMetadata> evaluate_existential(const JitTensorMetadata &in, const std::vector<size_t> &reduce_axes) {
//         if (!in.is_valid || reduce_axes.empty()) return std::nullopt;
//         std::vector to_remove(in.rank, false);
//         for (size_t axis: reduce_axes) {
//             if (axis >= in.rank) return std::nullopt;
//             to_remove[axis] = true;
//         }
//         std::vector<std::variant<std::string, uint64_t> > out_shape;
//         for (size_t i = 0; i < in.rank; ++i) {
//             if (!to_remove[i]) out_shape.push_back(in.shape[i]);
//         }
//         return JitTensorMetadata{TYPE_FLOAT, StorageLayout::Dense, out_shape.size(), out_shape, true};
//     }
//
//     // 23. EVALUATE_UNIVERSAL
//     static std::optional<JitTensorMetadata> evaluate_universal(const JitTensorMetadata &in, const std::vector<size_t> &reduce_axes) {
//         return evaluate_existential(in, reduce_axes);
//     }
//
//     // 24. AGGREGATE
//     static std::optional<JitTensorMetadata> aggregate(const JitTensorMetadata &in, const std::vector<size_t> &retained_axes) {
//         if (!in.is_valid || retained_axes.empty()) return std::nullopt;
//         std::vector<std::variant<std::string, uint64_t> > out_shape;
//         for (size_t axis: retained_axes) {
//             if (axis >= in.rank) return std::nullopt;
//             out_shape.push_back(in.shape[axis]);
//         }
//         return JitTensorMetadata{in.data_type, StorageLayout::Dense, out_shape.size(), out_shape, true};
//     }
// };


#endif // TENSORLIBRARY_JITASTNODES_H


