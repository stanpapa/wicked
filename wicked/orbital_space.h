#ifndef _wicked_orbital_space_h_
#define _wicked_orbital_space_h_

#include <map>
#include <memory>
#include <string>
#include <vector>

/// Type of orbital space
enum class RDMType {
  // Single creation/annihilation contractions that yields a Kronecker delta
  Occupied,
  // Single annihilation/creation contractions that yields a Kronecker delta
  Unoccupied,
  // Multi-leg contractions
  General
};

/// Spin types
enum class SpinType { SpinOrbital, SpinFree, Alpha, Beta };

class OrbitalSpaceInfo {
private:
  /// This type holds infomation about a space in a tuple
  /// (<label>, type of RDM, labels)
  using t_space_info = std::tuple<char, RDMType, std::vector<std::string>>;

public:
  OrbitalSpaceInfo();

  /// Set default spaces
  void default_spaces();

  /// Set default spaces
  void reset();

  /// Add an elementary space
  void add_space(char label, RDMType structure,
                 const std::vector<std::string> &indices);

  /// Return the number of elementary spaces
  int num_spaces() { return static_cast<int>(space_info_.size()); }

  /// The label of an orbital space
  char label(int pos) const;

  /// The label of an index that belongs to a given orbital space
  const std::string index_label(int pos, int idx) const;

  /// The structure of the density matrices for an orbital space
  RDMType dmstructure(int pos) const;

  /// The indices of an orbital space
  const std::vector<std::string> &indices(int pos) const;

  /// Maps a label into an orbital space
  int label_to_space(char label) const;

  /// return a string representation
  std::string str() const;

private:
  /// Vector of spaces
  std::vector<t_space_info> space_info_;

  /// Maps a space label to its index
  std::map<char, int> label_to_pos_;

  /// Maps orbital indices to a composite space
  std::map<std::string, int> indices_to_pos_;
};

extern std::shared_ptr<OrbitalSpaceInfo> osi;

std::shared_ptr<OrbitalSpaceInfo> get_osi();

/// Used to convert a string (e.g., "unoccupied") to a RDMType
RDMType string_to_rdmtype(const std::string &str);

#endif // _wicked_orbital_space_h_
