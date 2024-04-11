/*---------------------------------------------------------------------------*/
/*! \file
\brief particle wall result test for particle simulations
\level 2
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
#ifndef FOUR_C_PARTICLE_WALL_RESULT_TEST_HPP
#define FOUR_C_PARTICLE_WALL_RESULT_TEST_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_config.hpp"

#include "baci_lib_resulttest.hpp"
#include "baci_particle_engine_typedefs.hpp"

BACI_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace PARTICLEWALL
{
  class WallHandlerInterface;
}

namespace DRT
{
  class Discretization;
}

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace PARTICLEWALL
{
  /*!
   * \brief particle wall result test handler
   *
   * \author Sebastian Fuchs \date 03/2019
   */
  class WallResultTest final : public DRT::ResultTest
  {
   public:
    //! constructor
    explicit WallResultTest();

    /*!
     * \brief init wall result test
     *
     * \author Sebastian Fuchs \date 03/2019
     */
    void Init();

    /*!
     * \brief setup wall result test
     *
     * \author Sebastian Fuchs \date 03/2019
     *
     * \param[in] particleengineinterface interface to particle engine
     */
    void Setup(const std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface);

    /*!
     * \brief test node value
     *
     * \author Sebastian Fuchs \date 03/2019
     *
     * \param[in]  res        result line definition
     * \param[out] nerr       number of tests with errors
     * \param[out] test_count number of tests performed
     */
    void TestNode(INPUT::LineDefinition& res, int& nerr, int& test_count) override;

    /*!
     * \brief test special quantity
     *
     * \author Sebastian Fuchs \date 03/2019
     *
     * \param[in]  res        result line definition
     * \param[out] nerr       number of tests with errors
     * \param[out] test_count number of tests performed
     */
    void TestSpecial(INPUT::LineDefinition& res, int& nerr, int& test_count) override;

   private:
    //! interface to particle wall handler
    std::shared_ptr<PARTICLEWALL::WallHandlerInterface> particlewallinterface_;

    //! wall discretization
    Teuchos::RCP<const DRT::Discretization> walldiscretization_;
  };

}  // namespace PARTICLEWALL

/*---------------------------------------------------------------------------*/
BACI_NAMESPACE_CLOSE

#endif