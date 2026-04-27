// TRCC native extension: hot-path kernels in C++ via pybind11.
//
// Exposes two functions:
//   delta_and_parent(X, signals) -> (delta, parent)
//     For each point i, find the nearest point j with signals[j] > signals[i]
//     using a kd-tree expanding-radius search. The global max gets parent=-1
//     and delta = max distance to any other point.
//
//   propagate_labels(signals, parent, peaks) -> labels
//     Assign every point the label of its density parent, in descending order
//     of signal so parents are processed first.
//
// All distances are Euclidean. Inputs are float64, contiguous, row-major.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "third_party/nanoflann.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace py = pybind11;

// ---- nanoflann adaptor over a row-major (n, d) double matrix --------------

struct PointCloud {
    const double* data;
    size_t n;
    size_t d;

    inline size_t kdtree_get_point_count() const { return n; }
    inline double kdtree_get_pt(size_t idx, size_t dim) const {
        return data[idx * d + dim];
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>,
    PointCloud>;


// Convert a numpy 2D float64 array to a tightly-validated PointCloud.
static PointCloud make_cloud(const py::array_t<double, py::array::c_style | py::array::forcecast>& X) {
    if (X.ndim() != 2) throw std::invalid_argument("X must be 2D");
    PointCloud pc;
    pc.data = X.data();
    pc.n = static_cast<size_t>(X.shape(0));
    pc.d = static_cast<size_t>(X.shape(1));
    return pc;
}


// -------- delta_and_parent ---------------------------------------------------

py::tuple delta_and_parent(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> signals
) {
    PointCloud pc = make_cloud(X);
    const size_t n = pc.n;
    const size_t d = pc.d;
    if (signals.ndim() != 1 || (size_t)signals.shape(0) != n)
        throw std::invalid_argument("signals must be 1D length n");

    const double* sig = signals.data();

    // Build kd-tree
    KDTree tree(static_cast<int>(d), pc, nanoflann::KDTreeSingleIndexAdaptorParams(20));
    tree.buildIndex();

    py::array_t<double> delta(n);
    py::array_t<int64_t> parent(n);
    auto delta_mut = delta.mutable_unchecked<1>();
    auto parent_mut = parent.mutable_unchecked<1>();

    // Search strategy: progressively grow k until we find a higher-signal
    // neighbor; if k exceeds n we mark the global maximum.
    constexpr size_t kInit = 16;

    double global_max_dist = 0.0;
    int64_t global_max_idx = -1;
    double max_sig = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; ++i) {
        if (sig[i] > max_sig) { max_sig = sig[i]; global_max_idx = (int64_t)i; }
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (long long ii = 0; ii < (long long)n; ++ii) {
        size_t i = (size_t)ii;
        if ((int64_t)i == global_max_idx) {
            // assigned later
            continue;
        }
        const double s_i = sig[i];

        size_t k = std::min(kInit, n);
        bool found = false;
        double best_d2 = 0.0;
        int64_t best_j = -1;

        const double* query = pc.data + i * d;
        std::vector<size_t> idx_buf;
        std::vector<double> d_buf;

        while (true) {
            idx_buf.assign(k, 0);
            d_buf.assign(k, 0.0);
            nanoflann::KNNResultSet<double, size_t, size_t> rs(k);
            rs.init(idx_buf.data(), d_buf.data());
            tree.findNeighbors(rs, query);

            for (size_t r = 0; r < rs.size(); ++r) {
                size_t j = idx_buf[r];
                if (j == i) continue;
                if (sig[j] > s_i) {
                    best_d2 = d_buf[r];
                    best_j = (int64_t)j;
                    found = true;
                    break;
                }
            }
            if (found) break;
            if (k >= n) break;
            k = std::min(k * 4, n);
        }

        if (found) {
            delta_mut(i) = std::sqrt(best_d2);
            parent_mut(i) = best_j;
        } else {
            // No higher-signal point exists (ties); treat as a peak candidate.
            delta_mut(i) = 0.0;
            parent_mut(i) = -1;
        }
    }

    // Compute the max delta seen so we can give the global max a sentinel value.
    for (size_t i = 0; i < n; ++i) {
        if (delta_mut(i) > global_max_dist) global_max_dist = delta_mut(i);
    }
    if (global_max_idx >= 0) {
        delta_mut(global_max_idx) = (global_max_dist > 0.0) ? global_max_dist : 1.0;
        parent_mut(global_max_idx) = -1;
    }

    return py::make_tuple(delta, parent);
}


// -------- propagate_labels ---------------------------------------------------

py::array_t<int64_t> propagate_labels(
    py::array_t<double, py::array::c_style | py::array::forcecast> signals,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parent,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> peaks
) {
    const size_t n = (size_t)signals.shape(0);
    const double* sig = signals.data();
    const int64_t* par = parent.data();
    const int64_t* pk = peaks.data();
    const size_t kP = (size_t)peaks.shape(0);

    py::array_t<int64_t> labels(n);
    auto lab = labels.mutable_unchecked<1>();
    for (size_t i = 0; i < n; ++i) lab(i) = -1;
    for (size_t c = 0; c < kP; ++c) lab(pk[c]) = (int64_t)c;

    // Indices sorted by descending signal
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return sig[a] > sig[b]; });

    for (size_t k = 0; k < n; ++k) {
        size_t i = order[k];
        if (lab(i) != -1) continue;
        int64_t p = par[i];
        if (p < 0) {
            lab(i) = 0;  // global max with no peak label (rare edge case)
        } else {
            lab(i) = lab(p);
        }
    }
    return labels;
}


PYBIND11_MODULE(trcc_native, m) {
    m.doc() = "TRCC native hot-path kernels (pybind11 + nanoflann)";
    m.def("delta_and_parent", &delta_and_parent,
          py::arg("X"), py::arg("signals"),
          "For each point find nearest higher-signal point. "
          "Returns (delta, parent).");
    m.def("propagate_labels", &propagate_labels,
          py::arg("signals"), py::arg("parent"), py::arg("peaks"),
          "Propagate labels in descending-signal order via parent[].");
}
