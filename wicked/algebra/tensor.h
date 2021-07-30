#ifndef _wicked_tensor_h_
#define _wicked_tensor_h_

#include <string>
#include <vector>

#include "index.h"
#include "wicked-def.h"

/// This class represents a tensor labeled with orbital indices.
/// It holds information about the label and the indices of the tensor.
class Tensor {

public:
  // ==> Constructors <==
  explicit Tensor() {}

  Tensor(const std::string &label, const std::vector<Index> &lower,
         const std::vector<Index> &upper,
         SymmetryType symmetry = SymmetryType::Antisymmetric);

  // ==> Class public interface <==

  /// Return a reference to the label
  const std::string &label() const { return label_; }

  /// Return a reference to the lower indices
  const std::vector<Index> &lower() const { return lower_; }

  /// Return a reference to the upper indices
  const std::vector<Index> &upper() const { return upper_; }

  /// Set the lower indices
  void set_lower(const std::vector<Index> &indices) { lower_ = indices; }

  /// Set the upper indices
  void set_upper(const std::vector<Index> &indices) { upper_ = indices; }

  /// Return a vector containing all indices
  std::vector<Index> indices() const;

  /// Reindex this tensor
  void reindex(index_map_t &idx_map);

  /// Return the rank of the tensor
  int rank() const { return lower_.size() + upper_.size(); }

  /// Return the symmetry factor of this vector
  int symmetry_factor() const;

  /// Comparison operator used for sorting
  bool operator<(Tensor const &other) const;

  /// Comparison operator used for sorting
  bool operator==(Tensor const &other) const;

  /// Return a string representation
  std::string str() const;

  /// Return a LaTeX representation
  std::string latex() const;

  /// Return an ambit (C++ code) representation
  std::string ambit() const;

private:
  // ==> Class private data <==

  std::string label_;
  std::vector<Index> lower_;
  std::vector<Index> upper_;
  SymmetryType symmetry_;
};

/// A function to construct a tensor
Tensor make_tensor(const std::string &label,
                   const std::vector<std::string> &lower,
                   const std::vector<std::string> &upper,
                   SymmetryType symmetry);

/// Print to an output stream
std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

#endif // _wicked_tensor_h_
