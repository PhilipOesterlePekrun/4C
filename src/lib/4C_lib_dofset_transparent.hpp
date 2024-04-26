/*---------------------------------------------------------------------*/
/*! \file

\brief A set of degrees of freedom special for contact

\level 1


*/
/*---------------------------------------------------------------------*/

#ifndef FOUR_C_LIB_DOFSET_TRANSPARENT_HPP
#define FOUR_C_LIB_DOFSET_TRANSPARENT_HPP


#include "4C_config.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_lib_discret.hpp"
#include "4C_lib_dofset.hpp"

FOUR_C_NAMESPACE_OPEN


namespace DRT
{
  /// Alias dofset that shares dof numbers with another dofset
  /*!
  A special set of degrees of freedom, implemented in order to assign the same degrees of freedom to
  nodes belonging to two discretizations. This way two discretizations can assemble into the same
  position of the system matrix. As internal variable it holds a source discretization
  (Constructor). If such a nodeset is assigned to a sub-discretization, its dofs are assigned
  according to the dofs of the source.

  */
  class TransparentDofSet : public virtual DRT::DofSet
  {
   public:
    /*!
    \brief Standard Constructor
    */
    explicit TransparentDofSet(Teuchos::RCP<DRT::Discretization> sourcedis, bool parallel = false);



    /// create a copy of this object
    Teuchos::RCP<DofSet> Clone() override { return Teuchos::rcp(new TransparentDofSet(*this)); }

    /// Assign dof numbers to all elements and nodes of the discretization.
    int AssignDegreesOfFreedom(
        const DRT::Discretization& dis, const unsigned dspos, const int start) override;

    /// Assign dof numbers for new discretization using dof numbering from source discretization.
    void TransferDegreesOfFreedom(const DRT::Discretization& sourcedis,  ///< source discret
        const DRT::Discretization&
            newdis,      ///< discretization that gets dof numbering from source discret
        const int start  ///< offset for dof numbering (obsolete)
    );


    /// Assign dof numbers for new discretization using dof numbering from source discretization.
    /// for this version, newdis is allowed to be distributed completely different; the
    /// communication  of the dofs is done internally.
    void ParallelTransferDegreesOfFreedom(const DRT::Discretization& sourcedis,  ///< source discret
        const DRT::Discretization&
            newdis,      ///< discretization that gets dof numbering from source discret
        const int start  ///< offset for dof numbering (obsolete)
    );

    /// helper for ParallelTransferDegreesOfFreedom; unpack the received block to
    /// generate the current map node gid -> its dofs
    void UnpackLocalSourceDofs(
        std::map<int, std::vector<int>>& gid_to_dofs, std::vector<char>& rblock);

    /// helper for ParallelTransferDegreesOfFreedom; pack the current map
    /// node gid -> its dofs into a send block
    void PackLocalSourceDofs(
        std::map<int, std::vector<int>>& gid_to_dofs, CORE::COMM::PackBuffer& sblock);

    /// helper for ParallelTransferDegreesOfFreedom; add processor local information
    /// to the map unpack the received block to the current map node gid -> its dofs
    void SetSourceDofsAvailableOnThisProc(std::map<int, std::vector<int>>& gid_to_dofs);

    /// helper for ParallelTransferDegreesOfFreedom, an MPI send call
    void SendBlock(int numproc, int myrank, std::vector<char>& sblock,
        CORE::COMM::Exporter& exporter, MPI_Request& request);

    /// helper for ParallelTransferDegreesOfFreedom, an MPI receive call
    void ReceiveBlock(int numproc, int myrank, std::vector<char>& rblock,
        CORE::COMM::Exporter& exporter, MPI_Request& request);

   protected:
    Teuchos::RCP<DRT::Discretization> sourcedis_;  ///< source discretization

    bool parallel_;  ///< call ParallelTransferDegreesOfFreedom instead of TransferDegreesOfFreedom

  };  // class TransparentDofSet
}  // namespace DRT

FOUR_C_NAMESPACE_CLOSE

#endif