#include <algorithm>
#include <iostream>

#include "fmt/format.h"

#include "equation.h"
#include "expression.h"
#include "helpers/helpers.h"
#include "sqoperator.h"
#include "tensor.h"
#include "wicked-def.h"

Equation::Equation(const SymbolicTerm &lhs, const SymbolicTerm &rhs,
                   scalar_t factor)
    : lhs_(lhs), rhs_(rhs), factor_(factor) {}

const SymbolicTerm &Equation::lhs() const { return lhs_; }

const SymbolicTerm &Equation::rhs() const { return rhs_; }

scalar_t Equation::rhs_factor() const { return factor_; }

Expression Equation::rhs_expression() const {
  Expression expr;
  expr.add(rhs(), rhs_factor());
  return expr;
}

void Equation::set_summation_indices() {
  // todo: make it work with set/unordered_set
  // get all target indices
  std::vector<Index> target_indices;
  for (const Tensor& t : lhs_.tensors()) {
    if (t.indices().empty()) continue;
    for (const auto& i : t.indices()) {
      if (std::find(target_indices.begin(),target_indices.end(),i) != target_indices.end()) continue;
      target_indices.push_back(i);
    }
  }

  // if an index is not found in the list of target indices, it must mean it is summed
  for (auto& t : rhs_.tensors()) {
    for (auto& i : t.upper_mut()) {
      if (std::find(target_indices.begin(),target_indices.end(),i) != target_indices.end()) continue;
      i.is_summed(true);
    }
    for (auto& i : t.lower_mut()) {
      if (std::find(target_indices.begin(),target_indices.end(),i) != target_indices.end()) continue;
      i.is_summed(true);
    }
  }
  for (SQOperator& o : rhs_.ops()) {
      if (std::find(target_indices.begin(),target_indices.end(),o.index()) != target_indices.end()) continue;
      o.index_mut().is_summed(true);
  }
}


std::vector<Equation> Equation::expand_integrals_to_mulliken() {
  // check if there is a 2e integral present
  int pos = -1;
  for (int i = 0; i < rhs_.tensors().size(); i++) {
    if (rhs_.tensors()[i].label() != "V") continue;
    pos = i;
    break;
  }
  if (pos == -1) return {*this};

  // assume integrals only appear on the right-hand side

  // <pq||rs> -> (pr|qs)
  SymbolicTerm rhs = rhs_;
  Tensor V = rhs_.tensors()[pos];
  V.label_mut("I");
  Index r = V.lower()[0];
  V.lower_mut()[0] = V.upper()[1]; // p,q,q,s
  V.upper_mut()[1] = r;
  rhs.tensors()[pos] = V;
  Equation one = Equation(lhs_, rhs, factor_);

  // (pr|qs) -> (ps|qr)
  rhs = rhs_;
  V.upper_mut()[1] = V.lower()[1]; // p,s,q,s
  V.lower_mut()[1] = r;
  rhs.tensors()[pos] = V;
  Equation two = Equation(lhs_, rhs, factor_ * -1);

  return {one,two};
}

bool Equation::operator==(Equation const &other) const {
  return ((lhs() == other.lhs()) and (rhs() == other.rhs()) and
          (rhs_factor() == other.rhs_factor()));
}

std::string Equation::str() const {
  std::vector<std::string> str_vec;
  str_vec.push_back(lhs_.str());
  str_vec.push_back("+=");
  str_vec.push_back(factor_.str());
  str_vec.push_back(rhs_.str());
  return (join(str_vec, " "));
}

std::string Equation::str_age() const {
  // flag summation indices
  Equation tmp = *this;
  tmp.set_summation_indices();

  // expand <pq||rs> to (pr|qs) - (ps|qr)
  // todo: move out of str_age()
  auto expanded = tmp.expand_integrals_to_mulliken();

  std::vector<std::string> str_vec;
  str_vec.push_back(expanded[0].lhs().str_age());
  str_vec.push_back("+=");
  str_vec.push_back(expanded[0].rhs_factor().str_age());
  str_vec.push_back(expanded[0].rhs().str_age());
  if (expanded.size() == 1) return (join(str_vec, " "));

  std::vector<std::string> str_vec2;
  str_vec2.push_back(expanded[1].lhs().str_age());
  str_vec2.push_back("+=");
  str_vec2.push_back(expanded[1].rhs_factor().str_age());
  str_vec2.push_back(expanded[1].rhs().str_age());

  return (join(str_vec, " ") + "\n" + join(str_vec2, " "));
}

std::string Equation::latex() const { return str(); }

std::string get_unique_index(const std::string &s,
                             std::map<std::string, std::string> &index_map,
                             std::vector<char> &unused_indices);

std::string
get_unique_tensor_indices(const Tensor &t,
                          std::map<std::string, std::string> &index_map,
                          std::vector<char> &unused_indices);

std::string get_unique_index(const std::string &s,
                             std::map<std::string, std::string> &index_map,
                             std::vector<std::string> &unused_indices) {
  // is this index (something like "i" or "o2") in the map? If not, figure out
  // what it corresponds to.
  if (index_map.count(s) == 0) {
    // a character
    if (s.size() == 1) {
      if (std::find(unused_indices.begin(), unused_indices.end(), s) !=
          unused_indices.end()) {
        // if this character is unused, use it
        index_map[s] = s;
      } else {
        // if it is used, grab the first available index
        index_map[s] = unused_indices.back();
      }
    } else {
      index_map[s] = unused_indices.back();
    }
    // erase this character from the available characters to avoid reusing
    unused_indices.erase(
        std::remove(unused_indices.begin(), unused_indices.end(), index_map[s]),
        unused_indices.end());
  }
  return index_map[s];
}

std::string
get_unique_tensor_indices(const Tensor &t,
                          std::map<std::string, std::string> &index_map,
                          std::vector<std::string> &unused_indices) {
  std::string indices;
  for (const auto &l : t.upper()) {
    indices += get_unique_index(l.latex(), index_map, unused_indices);
  }
  for (const auto &l : t.lower()) {
    indices += get_unique_index(l.latex(), index_map, unused_indices);
  }
  return indices;
}

std::string Equation::compile(const std::string &format) const {
  if (format == "ambit") {
    std::vector<std::string> str_vec;
    str_vec.push_back(lhs_.compile(format) + " += " + factor_.compile(format));
    str_vec.push_back(rhs_.compile(format));
    return (join(str_vec, " * ") + ";");
  }

  if (format == "einsum") {
    std::vector<std::string> str_vec;
    const auto &lhs_tensor = lhs().tensors()[0];

    std::string lhs_tensor_label = lhs_tensor.label();
    for (const auto &l : lhs_tensor.upper()) {
      lhs_tensor_label += orbital_subspaces->label(l.space());
    }
    for (const auto &l : lhs_tensor.lower()) {
      lhs_tensor_label += orbital_subspaces->label(l.space());
    }

    str_vec.push_back(lhs_tensor_label +
                      " += " + fmt::format("{:.9f}", rhs_factor().to_double()) +
                      " * np.einsum(");

    std::map<std::string, std::string> index_map;
    std::vector<std::string> unused_indices = {
        "Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N",
        "M", "L", "K", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A",
        "z", "y", "x", "w", "v", "u", "t", "s", "r", "q", "p", "o", "n",
        "m", "l", "k", "j", "i", "h", "g", "f", "e", "d", "c", "b", "a"};

    std::vector<std::string> indices_vec;
    for (const auto &t : rhs().tensors()) {
      std::string tensor_indices =
          get_unique_tensor_indices(t, index_map, unused_indices);
      indices_vec.push_back(tensor_indices);
    }

    std::vector<std::string> args_vec;
    args_vec.push_back(
        "\"" + join(indices_vec, ",") + "->" +
        get_unique_tensor_indices(lhs_tensor, index_map, unused_indices) +
        "\"");
    for (const auto &t : rhs().tensors()) {
      std::string t_label = t.label() + "[\"";
      for (const auto &l : t.upper()) {
        t_label += orbital_subspaces->label(l.space());
      }
      for (const auto &l : t.lower()) {
        t_label += orbital_subspaces->label(l.space());
      }
      t_label += "\"]";
      args_vec.push_back(t_label);
    }
    str_vec.push_back(join(args_vec, ","));
    str_vec.push_back(",optimize=\"optimal\")");
    return join(str_vec, "");
  }
  std::string msg = "Equation::compile() - the argument '" + format +
                    "' is not valid. Choices are 'ambit' or 'einsum'";
  throw std::runtime_error(msg);
  return "";
}

std::ostream &operator<<(std::ostream &os, const Equation &eterm) {
  os << eterm.str();
  return os;
}
