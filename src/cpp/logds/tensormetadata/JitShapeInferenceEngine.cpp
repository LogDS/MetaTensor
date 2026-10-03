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
#include <logds/tensormetadata/JitShapeInferenceEngine.h>

std::string JitShapeInferenceEngine::find_symbol_root(const std::string &sym) const {
    std::string root = sym;
    while (symbol_aliases.find(root) != symbol_aliases.end() && symbol_aliases[root] != root) {
        root = symbol_aliases[root];
    }
    // Ottimizzazione del cammino
    std::string current = sym;
    while (symbol_aliases.find(current) != symbol_aliases.end() && symbol_aliases[current] != root) {
        std::string next = symbol_aliases[current];
        symbol_aliases[current] = root;
        current = next;
    }
    return root;
}

bool JitShapeInferenceEngine::dims_match(const std::variant<std::string, uint64_t> &d1,
    const std::variant<std::string, uint64_t> &d2) const {
    // Caso A: Entrambi valori numerici espliciti
    if (std::holds_alternative<uint64_t>(d1) && std::holds_alternative<uint64_t>(d2)) {
        return std::get<uint64_t>(d1) == std::get<uint64_t>(d2);
    }

    // Caso B: Entrambi simboli stringa (es: "Batch" e "N")
    if (std::holds_alternative<std::string>(d1) && std::holds_alternative<std::string>(d2)) {
        std::string s1_root = find_symbol_root(std::get<std::string>(d1));
        std::string s2_root = find_symbol_root(std::get<std::string>(d2));

        if (s1_root == s2_root) return true;

        // Se entrambi hanno vincoli numerici diversi, non possono essere unificati
        bool has_n1 = symbolic_numeric_bounds.find(s1_root) != symbolic_numeric_bounds.end();
        bool has_n2 = symbolic_numeric_bounds.find(s2_root) != symbolic_numeric_bounds.end();
        if (has_n1 && has_n2 && symbolic_numeric_bounds[s1_root] != symbolic_numeric_bounds[s2_root]) {
            return false;
        }

        // Unificazione dei due alberi simbolici: s1 diventa alias di s2
        symbol_aliases[s1_root] = s2_root;
        if (has_n1) symbolic_numeric_bounds[s2_root] = symbolic_numeric_bounds[s1_root];
        return true;
    }

    // Caso C: Scontro misto Simbolo-Numero (Unificazione con costante hardware)
    auto sym_part = std::holds_alternative<std::string>(d1) ? std::get<std::string>(d1) : std::get<std::string>(d2);
    auto num_part = std::holds_alternative<uint64_t>(d1) ? std::get<uint64_t>(d1) : std::get<uint64_t>(d2);
    std::string sym_root = find_symbol_root(sym_part);

    if (symbolic_numeric_bounds.find(sym_root) != symbolic_numeric_bounds.end()) {
        return symbolic_numeric_bounds[sym_root] == num_part;
    }

    // Vincoliamo permanentemente il simbolo al numero reale
    symbolic_numeric_bounds[sym_root] = num_part;
    return true;
}

bool JitShapeInferenceEngine::is_one(const std::variant<std::string, uint64_t> &d) const {
    if (std::holds_alternative<uint64_t>(d)) return std::get<uint64_t>(d) == 1;
    std::string root = find_symbol_root(std::get<std::string>(d));
    if (symbolic_numeric_bounds.find(root) != symbolic_numeric_bounds.end()) {
        return symbolic_numeric_bounds[root] == 1;
    }
    return false;
}

std::variant<std::string, uint64_t> JitShapeInferenceEngine::resolve_canonical_dimension(
    const std::variant<std::string, uint64_t> &d) const {
    if (std::holds_alternative<uint64_t>(d)) return d;
    std::string root = find_symbol_root(std::get<std::string>(d));
    if (symbolic_numeric_bounds.find(root) != symbolic_numeric_bounds.end()) {
        return symbolic_numeric_bounds[root];
    }
    return root;
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::permute_axes(const JitTensorMetadata &in,
    const std::vector<size_t> &perm) const {
    if (!in.is_valid || perm.size() != in.rank) return std::nullopt;
    std::vector<bool> checked(in.rank, false);
    std::vector<std::variant<std::string, uint64_t> > out_shape(in.rank);
    for (size_t i = 0; i < perm.size(); ++i) {
        size_t axis = perm[i];
        if (axis >= in.rank || checked[axis]) return std::nullopt;
        checked[axis] = true;
        out_shape[i] = in.shape[axis];
    }
    return JitTensorMetadata{in.data_type, in.layout, in.rank, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::reduce_sum(const JitTensorMetadata &in, size_t axis) const {
    if (!in.is_valid || in.rank == 0 || axis >= in.rank) return std::nullopt;
    std::vector<std::variant<std::string, uint64_t> > out_shape;
    for (size_t i = 0; i < in.rank; ++i) {
        if (i != axis) out_shape.push_back(in.shape[i]);
    }
    return JitTensorMetadata{in.data_type, in.layout, in.rank - 1, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::reduce_all_sum(const JitTensorMetadata &in) const {
    if (!in.is_valid) return std::nullopt;
    return make_scalar(in.data_type);
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::casting(const JitTensorMetadata &in,
    DataType target_type) const {
    if (!in.is_valid) return std::nullopt;
    return JitTensorMetadata{target_type, in.layout, in.rank, in.shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::float_extraction(const JitTensorMetadata &in) const {
    if (!in.is_valid) return std::nullopt;
    bool convertible = (in.rank == 0);
    if (in.rank > 0) {
        convertible = std::all_of(in.shape.begin(), in.shape.end(), [&](const auto &d) { return is_one(d); });
    }
    if (!convertible) return std::nullopt;
    return make_scalar(TYPE_FLOAT);
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::slice_tensor(const JitTensorMetadata &in, size_t axis,
    uint64_t start, uint64_t end) const {
    if (!in.is_valid || axis >= in.rank || end <= start) return std::nullopt;
    auto resolved_dim = resolve_canonical_dimension(in.shape[axis]);
    if (std::holds_alternative<uint64_t>(resolved_dim)) {
        if (end > std::get<uint64_t>(resolved_dim)) return std::nullopt;
    }
    std::vector<std::variant<std::string, uint64_t> > out_shape = in.shape;
    out_shape[axis] = end - start;
    return JitTensorMetadata{in.data_type, in.layout, in.rank, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::squeeze_axes(const JitTensorMetadata &in, size_t axis) const {
    if (!in.is_valid || axis >= in.rank || !is_one(in.shape[axis])) return std::nullopt;
    std::vector<std::variant<std::string, uint64_t> > out_shape;
    for (size_t i = 0; i < in.rank; ++i) {
        if (i != axis) out_shape.push_back(in.shape[i]);
    }
    return JitTensorMetadata{in.data_type, in.layout, in.rank - 1, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::segment_sum(const JitTensorMetadata &in,
    uint64_t num_segments) const {
    if (!in.is_valid || in.rank != 2) return std::nullopt;

    // Estraiamo la coordinata dell'asse 1 (le colonne), preservandone la natura simbolica o numerica
    auto retained_cols = in.shape[1];

    std::vector<std::variant<std::string, uint64_t>> out_shape = {
        std::variant<std::string, uint64_t>(num_segments),
        retained_cols
    };
    return JitTensorMetadata{in.data_type, in.layout, 2, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::one_hot_encoding(const JitTensorMetadata &in,
    uint64_t depth) const {
    if (!in.is_valid || in.rank != 1) return std::nullopt;

    // Estraiamo l'unico asse spaziale del vettore 1D originale
    auto rows_len = in.shape[0];

    std::vector<std::variant<std::string, uint64_t>> out_shape = {
        rows_len,
        std::variant<std::string, uint64_t>(depth)
    };
    return JitTensorMetadata{TYPE_FLOAT, StorageLayout::Dense, 2, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::operator_matmul(const JitTensorMetadata &left,
    const JitTensorMetadata &right) const {
    if (!left.is_valid || !right.is_valid || left.data_type != right.data_type || left.rank < 2 || right.rank < 2) {
        return std::nullopt;
    }

    // RISOLUTIVO: La contrazione avviene tra l'ULTIMO asse di sinistra ed il PRIMO asse di destra!
    if (!dims_match(left.shape[left.rank - 1], right.shape[0])) {
        return std::nullopt;
    }

    // Il tipo risultante eredita l'asse 0 di sinistra (M) e l'ultimo asse di destra (N)
    std::vector<std::variant<std::string, uint64_t>> out_shape = {
        resolve_canonical_dimension(left.shape[0]),
        resolve_canonical_dimension(right.shape[right.rank - 1])
    };
    return JitTensorMetadata{left.data_type, StorageLayout::Dense, 2, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::operator_cross(const JitTensorMetadata &left,
    const JitTensorMetadata &right) const {
    if (!left.is_valid || !right.is_valid || left.rank != 1 || right.rank != 1) {
        return std::nullopt;
    }

    // RISOLUTIVO: Verifichiamo l'indice zero (l'unico asse presente nei vettori 1D)
    if (!dims_match(left.shape[0], uint64_t(3)) || !dims_match(right.shape[0], uint64_t(3))) {
        return std::nullopt;
    }

    std::vector<std::variant<std::string, uint64_t>> out_shape = { uint64_t(3) };
    return JitTensorMetadata{left.data_type, StorageLayout::Dense, 1, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::gather_nd(const JitTensorMetadata &source,
    const JitTensorMetadata &indices) const {
    if (!source.is_valid || !indices.is_valid || source.rank != 2 || indices.rank != 2) return std::nullopt;
    return JitTensorMetadata{source.data_type, StorageLayout::Dense, 1, {indices.shape}, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::tensor_theta_join(const JitTensorMetadata &left,
    const JitTensorMetadata &right) const {
    if (!left.is_valid || !right.is_valid || left.rank != 1 || right.rank != 1) return std::nullopt;

    // RISOLUTIVO: Estraiamo l'unico asse reale indicizzandolo esplicitamente all'indice [0]
    auto l_dim = resolve_canonical_dimension(left.shape[0]);
    auto r_dim = resolve_canonical_dimension(right.shape[0]);

    std::variant<std::string, uint64_t> worst_case_len;
    if (std::holds_alternative<uint64_t>(l_dim) && std::holds_alternative<uint64_t>(r_dim)) {
        worst_case_len = std::get<uint64_t>(l_dim) * std::get<uint64_t>(r_dim);
    } else {
        std::string l_sym = std::holds_alternative<std::string>(l_dim)
                                ? std::get<std::string>(l_dim)
                                : std::to_string(std::get<uint64_t>(l_dim));
        std::string r_sym = std::holds_alternative<std::string>(r_dim)
                                ? std::get<std::string>(r_dim)
                                : std::to_string(std::get<uint64_t>(r_dim));
        worst_case_len = l_sym + "x" + r_sym;
    }

    // Il record finale descrive una matrice bidimensionale a due colonne (Coppie di coordinate i, j)
    std::vector<std::variant<std::string, uint64_t>> out_shape = {
        worst_case_len,
        std::variant<std::string, uint64_t>(uint64_t(2))
    };
    return JitTensorMetadata{TYPE_FLOAT, StorageLayout::Dense, 2, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::operator_add(const JitTensorMetadata &left,
    const JitTensorMetadata &right) const {
    if (!left.is_valid || !right.is_valid || left.data_type != right.data_type) return std::nullopt;
    if (left.rank == 0) return right;
    if (right.rank == 0) return left;
    size_t max_rank = std::max(left.rank, right.rank);
    std::vector<std::variant<std::string, uint64_t> > out_shape(max_rank);
    for (size_t i = 0; i < max_rank; ++i) {
        size_t l_idx = left.rank - 1 - i;
        size_t r_idx = right.rank - 1 - i;
        auto l_dim = (i < left.rank) ? left.shape[l_idx] : std::variant<std::string, uint64_t>(uint64_t(1));
        auto r_dim = (i < right.rank) ? right.shape[r_idx] : std::variant<std::string, uint64_t>(uint64_t(1));
        // RISOLUTIVO: dims_match esegue l'unificazione contestuale in-memory dei simboli!
        if (dims_match(l_dim, r_dim)) { out_shape[max_rank - 1 - i] = resolve_canonical_dimension(l_dim); } else if
        (is_one(l_dim)) { out_shape[max_rank - 1 - i] = resolve_canonical_dimension(r_dim); } else if (
            is_one(r_dim)) { out_shape[max_rank - 1 - i] = resolve_canonical_dimension(l_dim); } else return
                std::nullopt;
    }
    StorageLayout out_layout = (left.layout == StorageLayout::SparseCOO && right.layout == StorageLayout::SparseCOO)
                                   ? StorageLayout::SparseCOO
                                   : StorageLayout::Dense;
    return JitTensorMetadata{left.data_type, out_layout, max_rank, out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::contraction(const JitTensorMetadata &left,
    const JitTensorMetadata &right, size_t left_axis, size_t right_axis) const {
    if (!left.is_valid || !right.is_valid || left_axis >= left.rank || right_axis >= right.rank) return
            std::nullopt;
    if (!dims_match(left.shape[left_axis], right.shape[right_axis])) return std::nullopt;
    std::vector<std::variant<std::string, uint64_t> > out_shape;
    for (size_t i = 0; i < left.rank; ++i) if (i != left_axis) out_shape.push_back(
        resolve_canonical_dimension(left.shape[i]));
    for (size_t i = 0; i < right.rank; ++i) if (i != right_axis) out_shape.push_back(
        resolve_canonical_dimension(right.shape[i]));
    return JitTensorMetadata{left.data_type, StorageLayout::Dense, out_shape.size(), out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::where(const JitTensorMetadata &in, float off_value) const {
    if (!in.is_valid) return std::nullopt;
    StorageLayout out_layout = (off_value == 0.0f) ? in.layout : StorageLayout::Dense;
    return JitTensorMetadata{in.data_type, out_layout, in.rank, in.shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::operator_sub(const JitTensorMetadata &left,
    const JitTensorMetadata &right) const {
    return operator_add(left, right);
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::element_wise_mul(const JitTensorMetadata &left,
    const JitTensorMetadata &right) const {
    auto out = operator_add(left, right);
    if (!out) return std::nullopt;
    if (left.layout == StorageLayout::SparseCOO || right.layout == StorageLayout::SparseCOO) {
        out->layout = StorageLayout::SparseCOO;
    }
    return out;
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::operator_brackets(const JitTensorMetadata &in,
    const std::vector<uint64_t> &coords) const {
    if (!in.is_valid || coords.size() != in.rank) return std::nullopt;
    for (size_t i = 0; i < in.rank; ++i) {
        auto resolved_dim = resolve_canonical_dimension(in.shape[i]);
        if (std::holds_alternative<uint64_t>(resolved_dim)) {
            if (coords[i] >= std::get<uint64_t>(resolved_dim)) return std::nullopt;
        }
    }
    return make_scalar(in.data_type);
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::apply(const JitTensorMetadata &in,
    const std::string &op_name) const {
    if (!in.is_valid) return std::nullopt;
    bool preserves_zero = (op_name == "Abs" || op_name == "Sqrt" || op_name == "Square" || op_name == "Tanh");
    StorageLayout out_layout = preserves_zero ? in.layout : StorageLayout::Dense;
    return JitTensorMetadata{in.data_type, out_layout, in.rank, in.shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::evaluate_existential(const JitTensorMetadata &in,
    const std::vector<size_t> &reduce_axes) const {
    if (!in.is_valid || reduce_axes.empty()) return std::nullopt;
    std::vector to_remove(in.rank, false);
    for (size_t axis: reduce_axes) {
        if (axis >= in.rank) return std::nullopt;
        to_remove[axis] = true;
    }
    std::vector<std::variant<std::string, uint64_t> > out_shape;
    for (size_t i = 0; i < in.rank; ++i) {
        if (!to_remove[i]) out_shape.push_back(resolve_canonical_dimension(in.shape[i]));
    }
    return JitTensorMetadata{TYPE_FLOAT, StorageLayout::Dense, out_shape.size(), out_shape, true};
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::evaluate_universal(const JitTensorMetadata &in,
    const std::vector<size_t> &reduce_axes) const {
    return evaluate_existential(in, reduce_axes);
}

std::optional<JitTensorMetadata> JitShapeInferenceEngine::aggregate(const JitTensorMetadata &in,
    const std::vector<size_t> &retained_axes) const {
    if (!in.is_valid || retained_axes.empty()) return std::nullopt;
    std::vector<std::variant<std::string, uint64_t> > out_shape;
    for (size_t axis: retained_axes) {
        if (axis >= in.rank) return std::nullopt;
        out_shape.push_back(resolve_canonical_dimension(in.shape[axis]));
    }
    return JitTensorMetadata{in.data_type, StorageLayout::Dense, out_shape.size(), out_shape, true};
}
