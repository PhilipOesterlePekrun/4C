/*----------------------------------------------------------------------*/
/*! \file

\brief Routines for ScaTraHDG element

Scatra element based on the hybridizable discontinuous Galerkin method instead
of the usual Lagrangian polynomials for standard transport elements

\level 3


*/
#ifndef FOUR_C_SCATRA_ELE_HDG_HPP
#define FOUR_C_SCATRA_ELE_HDG_HPP

#include "4C_config.hpp"

#include "4C_discretization_fem_general_utils_polynomial.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_scatra_ele.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace INPUT
{
  class LineDefinition;
}

namespace DRT
{
  class Discretization;

  namespace ELEMENTS
  {
    class ScaTraHDGType : public TransportType
    {
     public:
      std::string Name() const override { return "ScaTraHDGType"; }

      static ScaTraHDGType& Instance();

      CORE::COMM::ParObject* Create(const std::vector<char>& data) override;

      Teuchos::RCP<DRT::Element> Create(const std::string eletype, const std::string eledistype,
          const int id, const int owner) override;

      Teuchos::RCP<DRT::Element> Create(const int id, const int owner) override;

      void NodalBlockInformation(Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      CORE::LINALG::SerialDenseMatrix ComputeNullSpace(
          DRT::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void SetupElementDefinition(
          std::map<std::string, std::map<std::string, INPUT::LineDefinition>>& definitions)
          override;

     private:
      static ScaTraHDGType instance_;
    };


    /*!
    \brief HDG transport element
    */
    class ScaTraHDG : public Transport
    {
     public:
      //! @name constructors and destructors and related methods

      /*!
      \brief standard constructor
      */
      ScaTraHDG(int id,  //!< A unique global id
          int owner      //!< ???
      );

      //! Makes a deep copy of a Element
      ScaTraHDG(const ScaTraHDG& old);

      /*!
      \brief Deep copy this instance of scatra and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      DRT::Element* Clone() const override;


      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int UniqueParObjectId() const override
      {
        return ScaTraHDGType::Instance().UniqueParObjectId();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element
      */
      void Unpack(const std::vector<char>& data) override;

      /*!
      \brief Pack Material

      \ref PackMaterial and \ref UnpackMaterial are used to adapt material evaluation

      */
      virtual void PackMaterial(CORE::COMM::PackBuffer& data) const;

      /*!
      \brief Unpack Material

      \ref PackMaterial and \ref UnpackMaterial are used to adapt material evaluation

      */
      virtual void UnpackMaterial(const std::vector<char>& data) const;

      //! initialize the element
      int Initialize() override;

      /*!
      \brief Read input for this element
      */
      bool ReadElement(const std::string& eletype, const std::string& distype,
          INPUT::LineDefinition* linedef) override;

      //@}

      //! @name Access methods
      /*!
      \brief Get number of degrees of freedom per node

      HDG element: No dofs are associated with nodes
      */
      int NumDofPerNode(const DRT::Node&) const override { return 0; }

      //! Get number of degrees of freedom per face
      int NumDofPerFace(const unsigned face) const override { return NumDofPerComponent(face); }

      //! Get number of dofs per component per face
      int NumDofPerComponent(const unsigned face) const override
      {
        return CORE::FE::getBasisSize(CORE::FE::getEleFaceShapeType(this->distype_),
            (this->Faces()[face])->Degree(), completepol_);
      }

      //! Get number of degrees of freedom per element, zero for the primary dof set
      //! and equal to the given number for the secondary dof set
      int NumDofPerElement() const override { return 0; }

      //! Returns the degree of the element
      int Degree() const override { return degree_; }

      //! Returns the degree of the element
      int DegreeOld() const { return degree_old_; }

      //! Sets the degree of the element
      void SetDegree(int degree)
      {
        degree_old_ = degree_;
        degree_ = degree;
        return;
      }

      //! Sets the number of all interior dofs of the element
      void SetDofs(int ndofs) { ndofs_ = ndofs; }

      //! Sets the number of all dofs on faces of the element
      void SetOnfDofs(int onfdofs)
      {
        onfdofs_old_ = onfdofs_;
        onfdofs_ = onfdofs;
      }

      //! Set completepol_ variable
      void SetCompletePolynomialSpace(bool completepol) { completepol_ = completepol; }

      //! Returns the degree of the element
      int UsesCompletePolynomialSpace() const { return completepol_; }

      //! Sets bool to false if degree of element changes after p-adaption
      void SetPadaptEle(bool adapt)
      {
        padpatele_ = adapt;
        return;
      }

      //! Returns bool if degree of element changes after p-adaption (true if degree changes)
      bool PadaptEle() const { return padpatele_; }

      //! Sets bool if element matrices are initialized
      void SetMatInit(bool matinit)
      {
        matinit_ = matinit;
        return;
      }

      //! Returns bool if element matrices are initialized
      bool MatInit() const { return matinit_; }

      //! Returns the degree of the element for the interior DG space
      int NumDofPerElementAuxiliary() const
      {
        return (CORE::FE::getDimension(distype_) + 1) *
               CORE::FE::getBasisSize(distype_, degree_, completepol_);
      }

      //! Get vector of Teuchos::RCPs to the lines of this element
      std::vector<Teuchos::RCP<DRT::Element>> Lines() override;

      //! Get vector of Teuchos::RCPs to the surfaces of this element
      std::vector<Teuchos::RCP<DRT::Element>> Surfaces() override;

      //! Get Teuchos::RCP to the internal face adjacent to this element as master element and the
      //! parent_slave element
      Teuchos::RCP<DRT::Element> CreateFaceElement(
          DRT::Element* parent_slave,  //!< parent slave fluid3 element
          int nnode,                   //!< number of surface nodes
          const int* nodeids,          //!< node ids of surface element
          DRT::Node** nodes,           //!< nodes of surface element
          const int lsurface_master,   //!< local surface number w.r.t master parent element
          const int lsurface_slave,    //!< local surface number w.r.t slave parent element
          const std::vector<int>& localtrafomap  //! local trafo map
          ) override;
      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate an element, that is, call the element routines to evaluate scatra
      element matrices and vectors or evaluate errors, statistics or updates etc. directly.

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controlling method does not expect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not expect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int Evaluate(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          LocationArray& la, CORE::LINALG::SerialDenseMatrix& elemat1,
          CORE::LINALG::SerialDenseMatrix& elemat2, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseVector& elevec2,
          CORE::LINALG::SerialDenseVector& elevec3) override;

      //@}

      int EvaluateNeumann(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          DRT::Condition& condition, std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseMatrix* elemat1) override
      {
        FOUR_C_THROW("Volume Neumann not implemented");
        return 0;
      }

      //! Print this element
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override { return ScaTraHDGType::Instance(); }

      // element matrices are stored to save calculation time, since they stay the same in the pure
      // diffusion reaction problem If necessary or worthwhile this can be changed and thus the
      // element matrices are calculated on the fly
      //! @name element matrices
      //!@{
      CORE::LINALG::SerialDenseMatrix Amat_;  //!< concentrations - concentrations
      CORE::LINALG::SerialDenseMatrix Bmat_;  //!< concentrations - concentrations gradients
      CORE::LINALG::SerialDenseMatrix Cmat_;  //!< concentration - trace
      CORE::LINALG::SerialDenseMatrix
          Dmat_;  //!< concentrations gradients - concentrations gradients
      CORE::LINALG::SerialDenseMatrix Emat_;    //!< trace - concentrations gradients
      CORE::LINALG::SerialDenseMatrix Gmat_;    //!< concentrations gradients
      CORE::LINALG::SerialDenseMatrix Hmat_;    //!< trace -trace
      CORE::LINALG::SerialDenseMatrix Mmat_;    //!< mass matrix (concentrations - concentrations)
      CORE::LINALG::SerialDenseMatrix EmatT_;   //!< trace - concentrations gradients (E^T)
      CORE::LINALG::SerialDenseMatrix BmatMT_;  //!< concentrations gradients- concentrations (-B^T)

      CORE::LINALG::SerialDenseMatrix Kmat_;      //!< condensed matrix
      CORE::LINALG::SerialDenseMatrix invAMmat_;  //!< inverse of [A + (1/(dt*theta))*M]
      //!@}

      //!@{ auxiliary stuff to store
      CORE::LINALG::SerialDenseMatrix BTAMmat_;
      CORE::LINALG::SerialDenseMatrix invCondmat_;
      CORE::LINALG::SerialDenseMatrix Xmat_;
      CORE::LINALG::SerialDenseVector Ivecnp_;
      CORE::LINALG::SerialDenseVector Ivecn_;
      CORE::LINALG::SerialDenseMatrix Imatnpderiv_;
      //!@}

      //! diffusion tensor
      CORE::LINALG::SerialDenseMatrix diff_;
      //! inverse diffusion tensor
      std::vector<CORE::LINALG::SerialDenseMatrix> invdiff_;
      //! main diffusivity
      double diff1_;


      //! stores the number of all interior dofs of the element
      unsigned int ndofs_;

      //! stores the number of all dofs for all faces
      unsigned int onfdofs_;

      //! stores the number of all old dofs for all faces
      unsigned int onfdofs_old_;


     private:
      //! don't want = operator
      ScaTraHDG& operator=(const ScaTraHDG& old);

      //! stores the degree of the element
      int degree_;

      //! stores the degree of the element
      int degree_old_;

      //! stores the polynomial type (tensor product or complete polynomial)
      bool completepol_;

      //! stores if element degree changes after p-adaption
      bool padpatele_;

      //! stores if element matrices are initialized
      bool matinit_;

    };  // class ScaTraHDG


    /*!
    \brief An element representing a boundary element of an scatrahdg element

    \note This is a pure Neumann boundary condition element. It's only
          purpose is to evaluate surface Neumann boundary conditions that might be
          adjacent to a parent scatrahdg element. It therefore does not implement
          the DRT::Element::Evaluate method and does not have its own ElementRegister class.

    */

    class ScaTraHDGBoundaryType : public DRT::ElementType
    {
     public:
      std::string Name() const override { return "ScaTraHDGBoundaryType"; }

      static ScaTraHDGBoundaryType& Instance();

      Teuchos::RCP<DRT::Element> Create(const int id, const int owner) override;

      void NodalBlockInformation(Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      CORE::LINALG::SerialDenseMatrix ComputeNullSpace(
          DRT::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        CORE::LINALG::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static ScaTraHDGBoundaryType instance_;
    };

    // class ScaTraHDGBoundary

    class ScaTraHDGBoundary : public DRT::FaceElement
    {
     public:
      //! @name Constructors and destructors and related methods

      //! number of space dimensions
      /*!
      \brief Standard Constructor

      \param id : A unique global id
      \param owner: Processor owning this surface
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes from
      \param parent: The parent acou element of this surface
      \param lsurface: the local surface number of this surface w.r.t. the parent element
      */
      ScaTraHDGBoundary(int id, int owner, int nnode, const int* nodeids, DRT::Node** nodes,
          DRT::Element* parent, const int lsurface);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      ScaTraHDGBoundary(const ScaTraHDGBoundary& old);

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      DRT::Element* Clone() const override;

      /*!
      \brief Get shape type of element
      */
      CORE::FE::CellType Shape() const override;

      /*!
      \brief Return number of lines of this element
      */
      int NumLine() const override { return CORE::FE::getNumberOfElementLines(Shape()); }

      /*!
      \brief Return number of surfaces of this element
      */
      int NumSurface() const override { return CORE::FE::getNumberOfElementSurfaces(Shape()); }

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element
      */
      std::vector<Teuchos::RCP<DRT::Element>> Lines() override;

      /*!
      \brief Get vector of Teuchos::RCPs to the surfaces of this element
      */
      std::vector<Teuchos::RCP<DRT::Element>> Surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int UniqueParObjectId() const override
      {
        return ScaTraHDGBoundaryType::Instance().UniqueParObjectId();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element

      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element
      */
      void Unpack(const std::vector<char>& data) override;



      //@}

      //! @name Acess methods


      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual DRT::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const DRT::Node& node) const override
      {
        return ParentElement()->NumDofPerNode(node);
      }

      int NumDofPerElement() const override { return 0; }

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override { return ScaTraHDGBoundaryType::Instance(); }

      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate element

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controling method does not epxect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controling method does not epxect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int Evaluate(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          std::vector<int>& lm, CORE::LINALG::SerialDenseMatrix& elemat1,
          CORE::LINALG::SerialDenseMatrix& elemat2, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseVector& elevec2,
          CORE::LINALG::SerialDenseVector& elevec3) override;

      //@}

      //! @name Evaluate methods

      /*!
      \brief Evaluate Neumann boundary condition

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): reference to the underlying discretization
      \param condition (in)     : condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int EvaluateNeumann(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          DRT::Condition& condition, std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseMatrix* elemat1 = nullptr) override;

      //@}

      /*!
      \brief Return the location vector of this element

      The method computes degrees of freedom this element adresses.
      Degree of freedom ordering is as follows:<br>
      First all degrees of freedom of adjacent nodes are numbered in
      local nodal order, then the element internal degrees of freedom are
      given if present.<br>
      If a derived element has to use a different ordering scheme,
      it is welcome to overload this method as the assembly routines actually
      don't care as long as matrices and vectors evaluated by the element
      match the ordering, which is implicitly assumed.<br>
      Length of the output vector matches number of degrees of freedom
      exactly.<br>
      This version is intended to fill the LocationArray with the dofs
      the element will assemble into. In the standard case these dofs are
      the dofs of the element itself. For some special conditions (e.g.
      the weak dirichlet boundary condtion) a surface element will assemble
      into the dofs of a volume element.<br>

      \note The degrees of freedom returned are not neccessarily only nodal dofs.
            Depending on the element implementation, output might also include
            element dofs.

      \param dis (in)      : the discretization this element belongs to
      \param la (out)      : location data for all dofsets of the discretization
      \param doDirichlet (in): whether to get the Dirichlet flags
      \param condstring (in): Name of condition to be evaluated
      \param condstring (in):  List of parameters for use at element level
      */
      void LocationVector(const Discretization& dis, LocationArray& la, bool doDirichlet,
          const std::string& condstring, Teuchos::ParameterList& params) const override;

      /*!
      \brief Return the location vector of this element

      The method computes degrees of freedom this element adresses.
      Degree of freedom ordering is as follows:<br>
      First all degrees of freedom of adjacent nodes are numbered in
      local nodal order, then the element internal degrees of freedom are
      given if present.<br>
      If a derived element has to use a different ordering scheme,
      it is welcome to overload this method as the assembly routines actually
      don't care as long as matrices and vectors evaluated by the element
      match the ordering, which is implicitly assumed.<br>
      Length of the output vector matches number of degrees of freedom
      exactly.<br>

      \note The degrees of freedom returned are not neccessarily only nodal dofs.
            Depending on the element implementation, output might also include
            element dofs.

      \param dis (in)      : the discretization this element belongs to
      \param lm (out)      : vector of degrees of freedom adressed by this element
      \param lmowner (out) : vector of proc numbers indicating which dofs are owned
                             by which procs in a dof row map. Ordering
                             matches dofs in lm.

      */
      void LocationVector(const Discretization& dis, std::vector<int>& lm,
          std::vector<int>& lmowner, std::vector<int>& lmstride) const override;

     private:
      // don't want = operator
      ScaTraHDGBoundary& operator=(const ScaTraHDGBoundary& old);

    };  // class ScaTraHDGBoundary


    /*!
    \brief An element representing an internal face element between two ScaTraHDG elements
    */
    class ScaTraHDGIntFaceType : public DRT::ElementType
    {
     public:
      std::string Name() const override { return "ScaTraHDGIntFaceType"; }

      static ScaTraHDGIntFaceType& Instance();

      Teuchos::RCP<DRT::Element> Create(const int id, const int owner) override;

      void NodalBlockInformation(Element* dwele, int& numdf, int& dimns, int& nv, int& np) override
      {
      }

      CORE::LINALG::SerialDenseMatrix ComputeNullSpace(
          DRT::Node& node, const double* x0, const int numdof, const int dimnsp) override
      {
        CORE::LINALG::SerialDenseMatrix nullspace;
        FOUR_C_THROW("method ComputeNullSpace not implemented!");
        return nullspace;
      }

     private:
      static ScaTraHDGIntFaceType instance_;
    };


    // class ScaTraHDGIntFace

    class ScaTraHDGIntFace : public DRT::FaceElement
    {
     public:
      //! @name Constructors and destructors and related methods

      //! number of space dimensions
      /*!
      \brief Standard Constructor

      \param id: A unique global id
      \param owner: Processor owning this surface
      \param nnode: Number of nodes attached to this element
      \param nodeids: global ids of nodes attached to this element
      \param nodes: the discretizations map of nodes to build ptrs to nodes from
      \param master_parent: The master parent ScaTraHDG element of this surface
      \param slave_parent: The slave parent ScaTraHDG element of this surface
      \param lsurface_master: the local surface number of this surface w.r.t. the master parent
      element \param lsurface_slave: the local surface number of this surface w.r.t. the slave
      parent element \param localtrafomap: transformation map between the local coordinate systems
      of the face w.r.t the master parent element's face's coordinate system and the slave element's
      face's coordinate system
      */
      ScaTraHDGIntFace(int id, int owner, int nnode, const int* nodeids, DRT::Node** nodes,
          DRT::ELEMENTS::ScaTraHDG* parent_master, DRT::ELEMENTS::ScaTraHDG* parent_slave,
          const int lsurface_master, const int lsurface_slave,
          const std::vector<int> localtrafomap);

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element
      */
      ScaTraHDGIntFace(const ScaTraHDGIntFace& old);

      /*!
      \brief Deep copy this instance of an element and return pointer to the copy

      The Clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      DRT::Element* Clone() const override;

      /*!
      \brief Get shape type of element
      */
      CORE::FE::CellType Shape() const override;

      /*!
      \brief Return number of lines of this element
      */
      int NumLine() const override { return CORE::FE::getNumberOfElementLines(Shape()); }

      /*!
      \brief Return number of surfaces of this element
      */
      int NumSurface() const override { return CORE::FE::getNumberOfElementSurfaces(Shape()); }

      /*!
      \brief Get vector of Teuchos::RCPs to the lines of this element
      */
      std::vector<Teuchos::RCP<DRT::Element>> Lines() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      std::vector<Teuchos::RCP<DRT::Element>> Surfaces() override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of the parobject.H file.
      */
      int UniqueParObjectId() const override
      {
        return ScaTraHDGIntFaceType::Instance().UniqueParObjectId();
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref Pack and \ref Unpack are used to communicate this element
      */
      void Pack(CORE::COMM::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref Pack and \ref Unpack are used to communicate this element
      */
      void Unpack(const std::vector<char>& data) override;



      //@}

      //! @name Acess methods

      /*!
      \brief Get number of degrees of freedom of a certain node
             (implements pure virtual DRT::Element)

      The element decides how many degrees of freedom its nodes must have.
      As this may vary along a simulation, the element can redecide the
      number of degrees of freedom per node along the way for each of it's nodes
      separately.
      */
      int NumDofPerNode(const DRT::Node& node) const override
      {
        return std::max(
            ParentMasterElement()->NumDofPerNode(node), ParentSlaveElement()->NumDofPerNode(node));
      }

      /*!
      \brief Get number of degrees of freedom per element
             (implements pure virtual DRT::Element)

      The element decides how many element degrees of freedom it has.
      It can redecide along the way of a simulation.

      \note Element degrees of freedom mentioned here are dofs that are visible
            at the level of the total system of equations. Purely internal
            element dofs that are condensed internally should NOT be considered.
      */
      int NumDofPerElement() const override { return 0; }

      /*!
       \brief Returns the degree of the element
       */
      int Degree() const override { return degree_; }

      /*!
       \brief Returns the old degree of the element
       */
      virtual int DegreeOld() const { return degree_old_; }

      void SetDegree(int degree)
      {
        degree_old_ = degree_;
        degree_ = degree;
        return;
      }

      /*!
      \brief create the location vector for patch of master and slave element

      \note All dofs shared by master and slave element are contained only once. Dofs from interface
      nodes are also included.
      */
      void PatchLocationVector(DRT::Discretization& discretization,  //!< discretization
          std::vector<int>& nds_master,        //!< nodal dofset w.r.t master parent element
          std::vector<int>& nds_slave,         //!< nodal dofset w.r.t slave parent element
          std::vector<int>& patchlm,           //!< local map for gdof ids for patch of elements
          std::vector<int>& master_lm,         //!< local map for gdof ids for master element
          std::vector<int>& slave_lm,          //!< local map for gdof ids for slave element
          std::vector<int>& face_lm,           //!< local map for gdof ids for face element
          std::vector<int>& lm_masterToPatch,  //!< local map between lm_master and lm_patch
          std::vector<int>& lm_slaveToPatch,   //!< local map between lm_slave and lm_patch
          std::vector<int>& lm_faceToPatch,    //!< local map between lm_face and lm_patch
          std::vector<int>&
              lm_masterNodeToPatch,  //!< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch  //!< local map between slave nodes and nodes in patch
      );

      /*!
      \brief Print this element
      */
      void Print(std::ostream& os) const override;

      DRT::ElementType& ElementType() const override { return ScaTraHDGIntFaceType::Instance(); }

      //@}

      //! @name Evaluation

      /*!
      \brief Evaluate element

      \param params (in/out): ParameterList for communication between control routine
                              and elements
      \param elemat1 (out)  : matrix to be filled by element. If nullptr on input,
                              the controling method does not epxect the element to fill
                              this matrix.
      \param elemat2 (out)  : matrix to be filled by element. If nullptr on input,
                              the controling method does not epxect the element to fill
                              this matrix.
      \param elevec1 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec2 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \param elevec3 (out)  : vector to be filled by element. If nullptr on input,
                              the controlling method does not epxect the element
                              to fill this vector
      \return 0 if successful, negative otherwise
      */
      int Evaluate(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          std::vector<int>& lm, CORE::LINALG::SerialDenseMatrix& elemat1,
          CORE::LINALG::SerialDenseMatrix& elemat2, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseVector& elevec2,
          CORE::LINALG::SerialDenseVector& elevec3) override;

      //@}

      //! @name Evaluate methods

      /*!
      \brief Evaluate Neumann boundary condition

      \param params (in/out)    : ParameterList for communication between control routine
                                  and elements
      \param discretization (in): reference to the underlying discretization
      \param condition (in)     : condition to be evaluated
      \param lm (in)            : location vector of this element
      \param elevec1 (out)      : vector to be filled by element. If nullptr on input,

      \return 0 if successful, negative otherwise
      */
      int EvaluateNeumann(Teuchos::ParameterList& params, DRT::Discretization& discretization,
          DRT::Condition& condition, std::vector<int>& lm, CORE::LINALG::SerialDenseVector& elevec1,
          CORE::LINALG::SerialDenseMatrix* elemat1 = nullptr) override;

      /*!
      \brief return the master parent ScaTraHDG element
      */
      DRT::ELEMENTS::ScaTraHDG* ParentMasterElement() const
      {
        DRT::Element* parent = this->DRT::FaceElement::ParentMasterElement();
        // make sure the static cast below is really valid
        FOUR_C_ASSERT(dynamic_cast<DRT::ELEMENTS::ScaTraHDG*>(parent) != nullptr,
            "Master element is no ScaTraHDG element");
        return static_cast<DRT::ELEMENTS::ScaTraHDG*>(parent);
      }

      /*!
      \brief return the slave parent ScaTraHDG element
      */
      DRT::ELEMENTS::ScaTraHDG* ParentSlaveElement() const
      {
        DRT::Element* parent = this->DRT::FaceElement::ParentSlaveElement();
        // make sure the static cast below is really valid
        FOUR_C_ASSERT(dynamic_cast<DRT::ELEMENTS::ScaTraHDG*>(parent) != nullptr,
            "Slave element is no ScaTraHDG element");
        return static_cast<DRT::ELEMENTS::ScaTraHDG*>(parent);
      }

      //@}

     private:
      // don't want = operator
      ScaTraHDGIntFace& operator=(const ScaTraHDGIntFace& old);

      // degree of this face element
      int degree_;

      // old degree of this face element
      int degree_old_;

    };  // class ScaTraHDGIntFace



  }  // namespace ELEMENTS
}  // namespace DRT


FOUR_C_NAMESPACE_CLOSE

#endif