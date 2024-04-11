/*----------------------------------------------------------------------*/
/*! \file

\brief provides the basic functionality for cutting a mesh with a level set function


\level 2
 *------------------------------------------------------------------------------------------------*/

#ifndef FOUR_C_CUT_LEVELSETINTERSECTION_HPP
#define FOUR_C_CUT_LEVELSETINTERSECTION_HPP

#include "baci_config.hpp"

#include "baci_cut_parentintersection.hpp"

BACI_NAMESPACE_OPEN


namespace LINALG
{
  class SerialDenseMatrix;
}

namespace CORE::GEO
{
  namespace CUT
  {
    class Node;
    class Edge;
    class Side;
    class Element;

    /*!
    \brief Interface class for the level set cut.
    */
    class LevelSetIntersection : public virtual ParentIntersection
    {
      typedef ParentIntersection my;


     public:
      LevelSetIntersection(const Epetra_Comm& comm, bool create_side = true);

      /// constructur for LevelSetIntersecton class
      LevelSetIntersection(int myrank = -1, bool create_side = true);

      /** \brief add a side of the cut mesh and return the side-handle
       *
       * (e.g. quadratic side-handle for quadratic sides) */
      void AddCutSide(int levelset_sid);

      ///
      bool HasLSCuttingSide(int sid) { return true; /*return sid == side_->Id();*/ };

      /*========================================================================*/
      //! @name Cut functionality, routines
      /*========================================================================*/
      //! @{

      /*! \brief Performs the cut of the mesh with the level set
       *
       *  standard Cut routine for parallel Level Set Cut where dofsets and node
       *  positions have to be parallelized
       *
       *  \author winter
       *  \date 08/14  */
      void Cut_Mesh(bool screenoutput = false) override;

      /*! \brief Performs all the level set cut operations including find positions
       *  and triangulation. (Used for the test cases)
       *
       *  Standard Cut routine for two phase flow and combustion where dofsets and
       *  node positions have not to be computed, standard cut for cut_test (Only used
       *  for cut test)
       *
       *  \author winter
       *  \date 08/14  */
      void Cut(bool include_inner = true, bool screenoutput = false,
          INPAR::CUT::VCellGaussPts VCellGP = INPAR::CUT::VCellGaussPts_Tessellation);

      //! @}
      /*========================================================================*/
      //! @name Add functionality for elements
      /*========================================================================*/
      //! @{

      /** \brief add this background element if it is cut. (determined by level set)
       *
       * Which implies that the level set function of the element has values which
       * are positive and negative. */
      CORE::GEO::CUT::ElementHandle* AddElement(int eid, const std::vector<int>& nids,
          const CORE::LINALG::SerialDenseMatrix& xyz, CORE::FE::CellType distype, const double* lsv,
          const bool lsv_only_plus_domain = false, const bool& check_lsv = false);

      //! @}

     private:
      const Epetra_Comm& Comm() const
      {
        if (not comm_) dserror("Epetra communicator was not initialized!");

        return *comm_;
      }

     protected:
      /*========================================================================*/
      //! @name private class variables
      /*========================================================================*/
      //! @{
      Teuchos::RCP<Side> side_;

      const Epetra_Comm* comm_;

      //! @}
    };

  }  // namespace CUT
}  // namespace CORE::GEO

BACI_NAMESPACE_CLOSE

#endif