/*! \file
\brief Nonlinear Shell 7-parameter model Finite Element

\level 3
*/

#include "baci_shell7p_ele.H"

#include "baci_lib_discret.H"
#include "baci_lib_utils_factory.H"
#include "baci_mat_so3_material.H"
#include "baci_shell7p_ele_factory.H"
#include "baci_shell7p_ele_interface_serializable.H"
#include "baci_shell7p_line.H"
#include "baci_shell7p_utils.H"


DRT::ELEMENTS::Shell7pType DRT::ELEMENTS::Shell7pType::instance_;

DRT::ELEMENTS::Shell7pType& DRT::ELEMENTS::Shell7pType::Instance() { return instance_; }


Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Shell7pType::Create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SHELL7P") return Create(id, owner);
  return Teuchos::null;
}

Teuchos::RCP<DRT::Element> DRT::ELEMENTS::Shell7pType::Create(const int id, const int owner)
{
  return Teuchos::rcp(new DRT::ELEMENTS::Shell7p(id, owner));
}

DRT::ParObject* DRT::ELEMENTS::Shell7pType::Create(const std::vector<char>& data)
{
  auto* object = new DRT::ELEMENTS::Shell7p(-1, -1);
  object->Unpack(data);
  return object;
}

void DRT::ELEMENTS::Shell7pType::NodalBlockInformation(
    Element* dwele, int& numdf, int& dimns, int& nv, int& np)
{
  STR::UTILS::SHELL::NodalBlockInformationShell(dwele, numdf, dimns, nv, np);
}

CORE::LINALG::SerialDenseMatrix DRT::ELEMENTS::Shell7pType::ComputeNullSpace(
    DRT::Node& node, const double* x0, const int numdof, const int dimnsp)
{
  auto* shell = dynamic_cast<DRT::ELEMENTS::Shell7p*>(node.Elements()[0]);
  if (!shell) dserror("Cannot cast to Shell7p");
  int j;
  for (j = 0; j < shell->NumNode(); ++j)
    if (shell->Nodes()[j]->Id() == node.Id()) break;
  if (j == shell->NumNode()) dserror("Can't find matching node..!");
  double half_thickness = shell->GetThickness() / 2.0;

  // set director
  const CORE::LINALG::SerialDenseMatrix nodal_directors = shell->GetDirectors();
  CORE::LINALG::Matrix<SHELL::DETAIL::num_dim, 1> director(true);
  for (int dim = 0; dim < SHELL::DETAIL::num_dim; ++dim)
    director(dim, 0) = nodal_directors(j, dim) * half_thickness;

  return STR::UTILS::SHELL::ComputeShellNullSpace(node, x0, director);
}


void DRT::ELEMENTS::Shell7pType::SetupElementDefinition(
    std::map<std::string, std::map<std::string, DRT::INPUT::LineDefinition>>& definitions)
{
  std::map<std::string, DRT::INPUT::LineDefinition>& defsgeneral = definitions["SHELL7P"];

  defsgeneral["QUAD4"] = INPUT::LineDefinition::Builder()
                             .AddIntVector("QUAD4", 4)
                             .AddNamedInt("MAT")
                             .AddNamedDouble("THICK")
                             .AddNamedString("EAS")
                             .AddString("EAS2")
                             .AddString("EAS3")
                             .AddString("EAS4")
                             .AddString("EAS5")
                             .AddNamedDouble("SDC")
                             .AddOptionalTag("ANS")
                             .AddOptionalNamedDoubleVector("RAD", 3)
                             .AddOptionalNamedDoubleVector("AXI", 3)
                             .AddOptionalNamedDoubleVector("CIR", 3)
                             .AddOptionalNamedDoubleVector("FIBER1", 3)
                             .AddOptionalNamedDoubleVector("FIBER2", 3)
                             .AddOptionalNamedDoubleVector("FIBER3", 3)
                             .Build();

  defsgeneral["QUAD8"] = INPUT::LineDefinition::Builder()
                             .AddIntVector("QUAD8", 8)
                             .AddNamedInt("MAT")
                             .AddNamedDouble("THICK")
                             .AddNamedString("EAS")
                             .AddString("EAS2")
                             .AddString("EAS3")
                             .AddString("EAS4")
                             .AddString("EAS5")
                             .AddNamedDouble("SDC")
                             .AddOptionalTag("ANS")
                             .AddOptionalNamedDoubleVector("RAD", 3)
                             .AddOptionalNamedDoubleVector("AXI", 3)
                             .AddOptionalNamedDoubleVector("CIR", 3)
                             .AddOptionalNamedDoubleVector("FIBER1", 3)
                             .AddOptionalNamedDoubleVector("FIBER2", 3)
                             .AddOptionalNamedDoubleVector("FIBER3", 3)
                             .Build();

  defsgeneral["QUAD9"] = INPUT::LineDefinition::Builder()
                             .AddIntVector("QUAD9", 9)
                             .AddNamedInt("MAT")
                             .AddNamedDouble("THICK")
                             .AddNamedString("EAS")
                             .AddString("EAS2")
                             .AddString("EAS3")
                             .AddString("EAS4")
                             .AddString("EAS5")
                             .AddNamedDouble("SDC")
                             .AddOptionalTag("ANS")
                             .AddOptionalNamedDoubleVector("RAD", 3)
                             .AddOptionalNamedDoubleVector("AXI", 3)
                             .AddOptionalNamedDoubleVector("CIR", 3)
                             .AddOptionalNamedDoubleVector("FIBER1", 3)
                             .AddOptionalNamedDoubleVector("FIBER2", 3)
                             .AddOptionalNamedDoubleVector("FIBER3", 3)
                             .Build();

  defsgeneral["TRI3"] = INPUT::LineDefinition::Builder()
                            .AddIntVector("TRI3", 3)
                            .AddNamedInt("MAT")
                            .AddNamedDouble("THICK")
                            .AddNamedDouble("SDC")
                            .AddOptionalNamedDoubleVector("RAD", 3)
                            .AddOptionalNamedDoubleVector("AXI", 3)
                            .AddOptionalNamedDoubleVector("CIR", 3)
                            .AddOptionalNamedDoubleVector("FIBER1", 3)
                            .AddOptionalNamedDoubleVector("FIBER2", 3)
                            .AddOptionalNamedDoubleVector("FIBER3", 3)
                            .Build();

  defsgeneral["TRI6"] = INPUT::LineDefinition::Builder()
                            .AddIntVector("TRI6", 6)
                            .AddNamedInt("MAT")
                            .AddNamedDouble("THICK")
                            .AddNamedDouble("SDC")
                            .AddOptionalNamedDoubleVector("RAD", 3)
                            .AddOptionalNamedDoubleVector("AXI", 3)
                            .AddOptionalNamedDoubleVector("CIR", 3)
                            .AddOptionalNamedDoubleVector("FIBER1", 3)
                            .AddOptionalNamedDoubleVector("FIBER2", 3)
                            .AddOptionalNamedDoubleVector("FIBER3", 3)
                            .Build();
}

int DRT::ELEMENTS::Shell7pType::Initialize(DRT::Discretization& dis)
{
  STR::UTILS::SHELL::DIRECTOR::SetupShellElementDirectors(*this, dis);

  return 0;
}



DRT::ELEMENTS::Shell7p::Shell7p(const DRT::ELEMENTS::Shell7p& other)
    : DRT::Element(other),
      distype_(other.distype_),
      interface_ptr_(other.interface_ptr_),
      eletech_(other.eletech_),
      thickness_(other.thickness_),
      nodal_directors_(other.nodal_directors_),
      material_post_setup_(other.material_post_setup_)
{
  // reset shell calculation interface
  shell_interface_ = Shell7pFactory::ProvideShell7pCalculationInterface(other, other.eletech_);
}

DRT::ELEMENTS::Shell7p& ::DRT::ELEMENTS::Shell7p::operator=(const DRT::ELEMENTS::Shell7p& other)
{
  if (this == &other) return *this;
  DRT::Element::operator=(other);
  distype_ = other.distype_;
  interface_ptr_ = other.interface_ptr_;
  eletech_ = other.eletech_;
  thickness_ = other.thickness_;
  nodal_directors_ = other.nodal_directors_;
  material_post_setup_ = other.material_post_setup_;

  shell_interface_ = Shell7pFactory::ProvideShell7pCalculationInterface(other, other.eletech_);
  return *this;
}


DRT::Element* DRT::ELEMENTS::Shell7p::Clone() const { return new Shell7p(*this); }


int DRT::ELEMENTS::Shell7p::NumLine() const
{
  return CORE::DRT::UTILS::getNumberOfElementLines(distype_);
}


int DRT::ELEMENTS::Shell7p::NumSurface() const { return 1; }


void DRT::ELEMENTS::Shell7p::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm(data);
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data, type);
  // add base class Element
  DRT::Element::Pack(data);
  // discretization type
  AddtoPack(data, (int)distype_);
  // element technology
  AddtoPack(data, eletech_);
  // thickness in reference frame
  AddtoPack(data, thickness_);
  // nodal_directors
  AddtoPack(data, nodal_directors_);
  // Setup flag for material post setup
  data.AddtoPack(material_post_setup_);
  // optional data, e.g., EAS data, current thickness,..
  std::shared_ptr<SHELL::Serializable> serializable_interface =
      std::dynamic_pointer_cast<SHELL::Serializable>(shell_interface_);
  if (serializable_interface != nullptr) serializable_interface->Pack(data);
}


void DRT::ELEMENTS::Shell7p::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position, data, type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");
  // extract base class Element
  std::vector<char> basedata(0);
  ExtractfromPack(position, data, basedata);
  Element::Unpack(basedata);
  // discretization type
  distype_ = static_cast<DRT::Element::DiscretizationType>(ExtractInt(position, data));
  // element technology
  ExtractfromPack(position, data, eletech_);
  // thickness in reference frame
  ExtractfromPack(position, data, thickness_);
  // nodal directors
  ExtractfromPack(position, data, nodal_directors_);
  // Setup flag for material post setup
  DRT::ParObject::ExtractfromPack(position, data, material_post_setup_);
  // reset shell calculation interface
  shell_interface_ = Shell7pFactory::ProvideShell7pCalculationInterface(*this, eletech_);
  std::shared_ptr<SHELL::Serializable> serializable_interface =
      std::dynamic_pointer_cast<SHELL::Serializable>(shell_interface_);
  if (serializable_interface != nullptr) serializable_interface->Unpack(position, data);

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d", (int)data.size(), position);
}


Teuchos::RCP<MAT::So3Material> DRT::ELEMENTS::Shell7p::SolidMaterial(int nummat) const
{
  return Teuchos::rcp_dynamic_cast<MAT::So3Material>(DRT::Element::Material(nummat), true);
}


void DRT::ELEMENTS::Shell7p::SetParamsInterfacePtr(const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
  {
    interface_ptr_ = Teuchos::rcp_dynamic_cast<STR::ELEMENTS::ParamsInterface>(
        p.get<Teuchos::RCP<DRT::ELEMENTS::ParamsInterface>>("interface"));
  }
  else
  {
    interface_ptr_ = Teuchos::null;
  }
}


void DRT::ELEMENTS::Shell7p::VisNames(std::map<std::string, int>& names)
{
  std::string result_thickness = "thickness";
  names[result_thickness] = 1;
  SolidMaterial()->VisNames(names);
}  // VisNames()


bool DRT::ELEMENTS::Shell7p::VisData(const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (DRT::Element::VisData(name, data)) return true;

  shell_interface_->VisData(name, data);

  return SolidMaterial()->VisData(name, data, Id());

}  // VisData()


void DRT::ELEMENTS::Shell7p::Print(std::ostream& os) const
{
  os << "Shell7p ";
  os << " Discretization type: " << DRT::DistypeToString(distype_).c_str();
  Element::Print(os);
}


std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::Shell7p::Lines()
{
  return DRT::UTILS::ElementBoundaryFactory<Shell7pLine, Shell7p>(DRT::UTILS::buildLines, *this);
}


std::vector<Teuchos::RCP<DRT::Element>> DRT::ELEMENTS::Shell7p::Surfaces()
{
  return {Teuchos::rcpFromRef(*this)};
}

bool DRT::ELEMENTS::Shell7p::ReadElement(
    const std::string& eletype, const std::string& distype, DRT::INPUT::LineDefinition* linedef)
{
  STR::ELEMENTS::ShellData shell_data = {};

  // set discretization type
  distype_ = DRT::StringToDistype(distype);

  // set thickness in reference frame
  linedef->ExtractDouble("THICK", thickness_);
  if (thickness_ <= 0) dserror("Shell element thickness needs to be > 0");
  shell_data.thickness = thickness_;

  // extract number of EAS parameters for different locking types
  STR::ELEMENTS::ShellLockingTypes locking_types = {};
  if (linedef->HaveNamed("EAS"))
  {
    eletech_.insert(INPAR::STR::EleTech::eas);
    STR::UTILS::SHELL::READELEMENT::ReadAndSetLockingTypes(distype_, linedef, locking_types);
  }

  // set calculation interface pointer
  shell_interface_ = Shell7pFactory::ProvideShell7pCalculationInterface(*this, eletech_);

  // read and set ANS technology for element
  if (linedef->HaveNamed("ANS"))
  {
    shell_data.num_ans = STR::UTILS::SHELL::READELEMENT::ReadAndSetNumANS(distype_);
  }

  // read SDC
  linedef->ExtractDouble("SDC", shell_data.sdc);

  // read and set number of material model
  SetMaterial(STR::UTILS::SHELL::READELEMENT::ReadAndSetElementMaterial(linedef));

  // setup shell calculation interface
  shell_interface_->Setup(*this, *SolidMaterial(), linedef, locking_types, shell_data);
  if (!material_post_setup_)
  {
    shell_interface_->MaterialPostSetup(*this, *SolidMaterial());
    material_post_setup_ = true;
  }
  return true;
}