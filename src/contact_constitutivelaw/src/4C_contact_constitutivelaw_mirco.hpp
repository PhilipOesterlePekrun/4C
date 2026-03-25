// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_MIRCO_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_MIRCO_HPP

#include "4C_config.hpp"

#include "4C_contact_constitutivelaw_contactconstitutivelaw.hpp"
#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#ifdef FOUR_C_WITH_MIRCO
#include <mirco_inputparameters.h>
#endif

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  namespace CONSTITUTIVELAW
  {
    /*----------------------------------------------------------------------*/
    /** \brief constitutive law parameters for a mirco contact law to the contact pressure
     *
     */
    struct MircoConstitutiveLawParams : public Parameter, public MIRCO::InputParameters
    {
     public:
      /** \brief standard constructor
       * \param[in] container containing the law parameter from the input file
       */
      MircoConstitutiveLawParams(const Core::IO::InputParameterContainer& container);

      int first_mat_id;
      int second_mat_id;
      double finite_difference_fraction;
      double active_gap_tolerance;

      // private: using MIRCO::InputParameters:://#/# anything?
    };  // class

    /*----------------------------------------------------------------------*/
    /** \brief implements a mirco contact constitutive law relating the gap to the
     * contact pressure
     */
    class MircoConstitutiveLaw : public ConstitutiveLaw
    {
     public:
      /// construct the constitutive law object given a set of parameters
      explicit MircoConstitutiveLaw(CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams params);

      //! @name Access methods
      //@{

      /// Return quick accessible contact constitutive law parameter data
      const CONTACT::CONSTITUTIVELAW::Parameter* parameter() const override { return &params_; }

      //@}

      //! @name Evaluation methods
      //@{
      /** \brief Evaluate the constitutive law
       *
       * The pressure response for a gap is calucated using MIRCO, which uses BEM for solving
       * contact between a rigid rough surface and a linear elastic half space.
       *
       * \param gap contact gap at the mortar node
       * \return The pressure response from MIRCO
       */
      double evaluate(const double gap, CONTACT::Node* cnode) override;

      /** \brief Evaluate derivative of the constitutive law
       *
       * The derivative of the pressure response is approximated using a finite difference approach
       * by calling MIRCO twice at two different gap values and doing a backward difference
       * approximation for the linearization.
       *
       * \param gap contact gap at the mortar node
       * \return Derivative of the pressure responses from MIRCO
       */
      double evaluate_derivative(const double gap, CONTACT::Node* cnode) override;
      //@}

     private:
      /// my constitutive law parameters
      CONTACT::CONSTITUTIVELAW::MircoConstitutiveLawParams params_;
    };
  }  // namespace CONSTITUTIVELAW
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
