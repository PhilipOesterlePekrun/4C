/*----------------------------------------------------------------------*/
/*! \file

\brief class that provides the common functionality for a mesh cut based on a level set field or on
surface meshes

\level 3
*------------------------------------------------------------------------------------------------*/

#ifndef FOUR_C_CUT_CUTWIZARD_HPP
#define FOUR_C_CUT_CUTWIZARD_HPP

#include "baci_config.hpp"

#include "baci_inpar_cut.hpp"
#include "baci_linalg_fixedsizematrix.hpp"
#include "baci_linalg_serialdensematrix.hpp"
#include "baci_linalg_serialdensevector.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

BACI_NAMESPACE_OPEN

namespace DRT
{
  class Discretization;
  class Element;
}  // namespace DRT

namespace CORE::LINALG
{
  class SerialDenseMatrix;
}

namespace XFEM
{
  class ConditionManager;
}

namespace CORE::GEO
{
  namespace CUT
  {
    class CombIntersection;
    class ElementHandle;
    class Node;
    class SideHandle;
  }  // namespace CUT

  /// contains the cut, and shared functionality between the level set and mesh cut.
  class CutWizard
  {
   public:
    /*------------------------------------------------------------------------*/
    /*! \brief Container class for the background mesh object
     *
     *  \author hiermeier \date 01/17 */
    class BackMesh
    {
     public:
      /// constructor
      explicit BackMesh(
          const Teuchos::RCP<DRT::Discretization>& backdis, CORE::GEO::CutWizard* wizard)
          : wizard_(wizard),
            back_discret_(backdis),
            back_disp_col_(Teuchos::null),
            back_levelset_col_(Teuchos::null)
      {
        if (backdis.is_null()) dserror("null pointer to background dis, invalid!");
      }

      virtual ~BackMesh() = default;

      void Init(const Teuchos::RCP<const Epetra_Vector>& back_disp_col,
          const Teuchos::RCP<const Epetra_Vector>& back_levelset_col);

      const Teuchos::RCP<DRT::Discretization>& GetPtr() { return back_discret_; }

      DRT::Discretization& Get() { return *back_discret_; }

      const DRT::Discretization& Get() const { return *back_discret_; }

      virtual int NumMyColElements() const;

      virtual const DRT::Element* lColElement(int lid) const;

      inline bool IsBackDisp() const { return (not back_disp_col_.is_null()); }

      const Epetra_Vector& BackDispCol() const
      {
        if (not IsBackDisp()) dserror("The background displacement was not initialized correctly!");

        return *back_disp_col_;
      }

      inline bool IsLevelSet() const { return (not back_levelset_col_.is_null()); }

      const Epetra_Vector& BackLevelSetCol() const
      {
        if (not IsLevelSet()) dserror("No level-set values set for the background discretization!");

        return *back_levelset_col_;
      }


     protected:
      CORE::GEO::CutWizard* wizard_;

     private:
      /// background discretization
      Teuchos::RCP<DRT::Discretization> back_discret_;

      /// col vector holding background ALE displacements for backdis
      Teuchos::RCP<const Epetra_Vector> back_disp_col_;

      /// col vector holding nodal level-set values based on backdis
      Teuchos::RCP<const Epetra_Vector> back_levelset_col_;
    };

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Container class for a certain cutting mesh objects
     */
    class CutterMesh
    {
     public:
      //! ctor
      CutterMesh(Teuchos::RCP<DRT::Discretization> cutterdis,
          Teuchos::RCP<const Epetra_Vector> cutter_disp_col, const int start_ele_gid)
          : cutterdis_(cutterdis), cutter_disp_col_(cutter_disp_col), start_ele_gid_(start_ele_gid)
      {
      }

      //---------------------------------discretization-----------------------------

      //! @name cutter discretization
      Teuchos::RCP<DRT::Discretization> cutterdis_;  ///< cutter discretization
      //@}

      //---------------------------------state vectors ----------------------------

      //! @name state vectors holding displacements
      Teuchos::RCP<const Epetra_Vector>
          cutter_disp_col_;  ///< col vector holding interface displacements for cutterdis
      //@}

      //!
      int start_ele_gid_;
    };

    /*========================================================================*/
    //! @name Constructor and Destructor
    /*========================================================================*/

    /*!
    \brief Constructor
    */
    CutWizard(const Teuchos::RCP<DRT::Discretization>& backdis);


    /*!
    \brief Destructor
    */
    virtual ~CutWizard() = default;

    //@}

    /*========================================================================*/
    //! @name Setters
    /*========================================================================*/

    //! set options and flags used during the cut
    void SetOptions(INPAR::CUT::NodalDofSetStrategy
                        nodal_dofset_strategy,     //!< strategy for nodal dofset management
        INPAR::CUT::VCellGaussPts VCellgausstype,  //!< Gauss point generation method for Volumecell
        INPAR::CUT::BCellGaussPts
            BCellgausstype,  //!< Gauss point generation method for Boundarycell
        bool gmsh_output,    //!< print write gmsh output for cut
        bool positions,      //!< set inside and outside point, facet and volumecell positions
        bool tetcellsonly,   //!< generate only tet cells
        bool screenoutput    //!< print screen output
    );

    virtual void SetBackgroundState(
        Teuchos::RCP<const Epetra_Vector>
            back_disp_col,  //!< col vector holding background ALE displacements for backdis
        Teuchos::RCP<const Epetra_Vector>
            back_levelset_col,  //!< col vector holding nodal level-set values based on backdis
        int level_set_sid       //!< global id for level-set side
    );

    void AddCutterState(const int mc_idx, Teuchos::RCP<DRT::Discretization> cutter_dis,
        Teuchos::RCP<const Epetra_Vector> cutter_disp_col);

    void AddCutterState(const int mc_idx, Teuchos::RCP<DRT::Discretization> cutter_dis,
        Teuchos::RCP<const Epetra_Vector> cutter_disp_col, const int start_ele_gid);

    // Find marked background-boundary sides.
    //  Extract these sides and create boundary cell for these!
    void SetMarkedConditionSides(
        // const int mc_idx,
        Teuchos::RCP<DRT::Discretization> cutter_dis,
        // Teuchos::RCP<const Epetra_Vector> cutter_disp_col,
        const int start_ele_gid);

    //@}

    /*========================================================================*/
    //! @name main Cut call
    /*========================================================================*/

    //! prepare the cut, add background elements and cutting sides
    void Prepare();

    void Cut(bool include_inner  //!< perform cut in the interior of the cutting mesh
    );

    /*========================================================================*/
    //! @name Accessors
    /*========================================================================*/

    //! Get this side (not from cut meshes) (faces of background elements) from the cut libraries
    CORE::GEO::CUT::SideHandle* GetSide(std::vector<int>& nodeids);

    //! Get this side (not from cut meshes) from the cut libraries
    CORE::GEO::CUT::SideHandle* GetSide(int sid);

    //! Get this side from cut meshes from the cut libraries
    CORE::GEO::CUT::SideHandle* GetCutSide(int sid);

    //! Get this element from the cut libraries by element id
    CORE::GEO::CUT::ElementHandle* GetElement(const int eleid) const;

    //! Get this element from the cut libraries by element pointer
    CORE::GEO::CUT::ElementHandle* GetElement(const DRT::Element* ele) const;

    //! Get this node from the cut libraries
    CORE::GEO::CUT::Node* GetNode(int nid);

    //! Get the sidehandle for cutting sides
    CORE::GEO::CUT::SideHandle* GetMeshCuttingSide(int sid, int mi);

    //! is there a level-set side with the given sid?
    bool HasLSCuttingSide(int sid);

    //! update the coordinates of the cut boundary cells
    void UpdateBoundaryCellCoords(Teuchos::RCP<DRT::Discretization> cutterdis,
        Teuchos::RCP<const Epetra_Vector> cutter_disp_col, const int start_ele_gid);

    //! Cubaturedegree for creating of integrationpoints on boundarycells
    int Get_BC_Cubaturedegree() const;

   protected:
    /** \brief hidden constructor for derived classes only
     *
     *  \author hiermeier \date 01/17 */
    CutWizard(const Epetra_Comm& comm);

    Teuchos::RCP<BackMesh>& BackMeshPtr() { return back_mesh_; }

    Teuchos::RCP<const BackMesh> BackMeshPtr() const { return back_mesh_.getConst(); }

    virtual void GetPhysicalNodalCoordinates(
        const DRT::Element* element, CORE::LINALG::SerialDenseMatrix& xyze) const;

    CORE::GEO::CUT::CombIntersection& Intersection()
    {
      if (intersection_.is_null()) dserror("nullptr pointer!");

      return *intersection_;
    }

   private:
    /*========================================================================*/
    //! @name Add functionality for elements and cutting sides
    /*========================================================================*/

    //! add all cutting sides (mesh and level-set sides)
    void AddCuttingSides();

    //! add level-set cutting side
    void AddLSCuttingSide();

    //! add all cutting sides from the cut-discretization
    void AddMeshCuttingSide();

    //! add elements from the background discretization
    void AddBackgroundElements();

    //! Add all cutting side elements of given cutter discretization with given displacement field
    //! to the intersection class
    void AddMeshCuttingSide(Teuchos::RCP<DRT::Discretization> cutterdis,
        Teuchos::RCP<const Epetra_Vector> cutter_disp_col,
        const int start_ele_gid = 0  ///< global start index for element id numbering
    );

    //! Add this cutting side element with given global coordinates to the intersection class
    void AddMeshCuttingSide(int mi, DRT::Element* ele, const CORE::LINALG::SerialDenseMatrix& xyze,
        const int start_ele_gid);

    //! Add this background mesh element to the intersection class
    void AddElement(const DRT::Element* ele, const CORE::LINALG::SerialDenseMatrix& xyze,
        double* myphinp = nullptr, bool lsv_only_plus_domain = false);

    //@}


    /*========================================================================*/
    //! @name Major steps to prepare the cut, to perform it and to do postprocessing
    /*========================================================================*/

    //! perform the actual cut, the intersection
    void Run_Cut(bool include_inner  //!< perform cut in the interior of the cutting mesh
    );

    //! routine for finding node positions and computing volume-cell dofsets in a parallel way
    void FindPositionDofSets(bool include_inner);

    //! write statistics and output to screen and files
    void Output(bool include_inner);

    //! Check that cut is initialized correctly
    bool SafetyChecks(bool is_prepare_cut_call);

    //@}

    /*========================================================================*/
    //! @name Output routines
    /*========================================================================*/

    /*! Print the number of volumecells and boundarycells generated over the
     *  whole mesh during the cut */
    void PrintCellStats();

    //! Write the DOF details of the nodes
    void DumpGmshNumDOFSets(bool include_inner);

    //! Write volumecell output in GMSH format throughout the domain
    void DumpGmshVolumeCells(bool include_inner);

    //! Write the integrationcells and boundarycells in GMSH format throughout the domain
    void DumpGmshIntegrationCells();

    //@}

    //---------------------------------discretizations----------------------------

    //! @name meshes
    Teuchos::RCP<BackMesh> back_mesh_;
    std::map<int, Teuchos::RCP<CutterMesh>> cutter_meshes_;
    const Epetra_Comm& comm_;
    int myrank_;  ///< my processor Id
    //@}

    //---------------------------------main intersection class----------------------------
    //! @name main intersection class and flags
    Teuchos::RCP<CORE::GEO::CUT::CombIntersection>
        intersection_;  ///< combined intersection object which handles cutting mesh sides and a
                        ///< level-set side

    bool do_mesh_intersection_;      ///< flag to perform intersection with mesh sides
    bool do_levelset_intersection_;  ///< flag to perform intersection with a level-set side
    //@}

    //---------------------------------state vectors ----------------------------

    //! @name state vectors holding displacements and level-set values
    int level_set_sid_;
    //@}

    //---------------------------------Options ----------------------------

    //! @name Options
    INPAR::CUT::VCellGaussPts VCellgausstype_;  ///< integration type for volume-cells
    INPAR::CUT::BCellGaussPts BCellgausstype_;  ///< integration type for boundary-cells
    bool gmsh_output_;                          ///< write gmsh output?
    bool tetcellsonly_;          ///< enforce to create tetrahedral integration cells exclusively
    bool screenoutput_;          ///< write output to screen
    bool lsv_only_plus_domain_;  ///< consider only plus domain of level-set field as physical field
    //@}

    //--------------------------------- Initialization flags ----------------------------

    //! @name Flags whether wizard is initialized correctly
    bool is_set_options_;
    bool is_cut_prepare_performed_;
    //@}

  };  // class CutWizard
}  // namespace CORE::GEO

BACI_NAMESPACE_CLOSE

#endif