// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_CONDITION_DEFINITION_HPP
#define FOUR_C_FEM_CONDITION_DEFINITION_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec.hpp"

#include <Teuchos_Array.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO
{
  class InputFile;
}

namespace Core::Conditions
{

  //--------------------------------------------------------------

  /// definition of a valid condition in 4C input
  /*!

    A ConditionDefinition is the definition of a condition dat file
    section. This definition includes the knowledge what this section looks
    like, how to read it and how to write it. In particular given a
    ConditionDefinition object it is possible to (a) write an empty dat file
    section that describes this condition, (b) read a dat file and create
    Core::Conditions::Condition objects for each line in this section and (c) write the dat
    file section filled with all corresponding conditions from a given
    Core::FE::Discretization.

    So this is quite sophisticated internal stuff here. If you want to
    introduce a new condition to 4C, all you have to do is add an
    appropriate definition in valid_conditions(). This will take care of the
    reading part and you will get your Core::FE::Discretization filled with proper
    Core::Conditions::Condition objects.

    \author u.kue
    \date 01/08
   */
  class ConditionDefinition
  {
   public:
    /// construction of a condition definition
    /*!
      \param sectionname name of dat file section
      \param conditionname name of conditions in Core::FE::Discretization
      \param description description of condition type
      \param condtype type of conditions to be build
      \param buildgeometry whether we need conditions elements
      \param gtype type of geometry the condition lives on
     */
    ConditionDefinition(std::string sectionname, std::string conditionname, std::string description,
        Core::Conditions::ConditionType condtype, bool buildgeometry,
        Core::Conditions::GeometryType gtype);

    /// add a concrete component to the condition line definition
    /*!
      Add new components to the input line. One at a time. Form left to
      right. The order is important! On reading we try and read component
      after component.
     */
    void add_component(Core::IO::InputSpec&& spec);

    void add_component(const Core::IO::InputSpec& spec);

    /// read all conditions from my input file section
    /*!
      \param problem (i) global problem instance that manages the input
      \param input (i) the input file
      \param cmap (o) the conditions we read here
     */
    void read(Core::IO::InputFile& input,
        std::multimap<int, std::shared_ptr<Core::Conditions::Condition>>& cmap) const;

    /// print my dat file section
    std::ostream& print(std::ostream& stream);

    /// name of my section in input file
    std::string section_name() const { return sectionname_; }

    /// my condition name
    std::string name() const { return conditionname_; }

    /// my condition description
    std::string description() const { return description_; }

    /// my GeometryType
    Core::Conditions::GeometryType geometry_type() const { return gtype_; }

    const std::vector<Core::IO::InputSpec>& specs() const { return specs_; }

   private:
    std::string sectionname_;
    std::string conditionname_;
    std::string description_;
    Core::Conditions::ConditionType condtype_;
    bool buildgeometry_;
    Core::Conditions::GeometryType gtype_;

    std::vector<Core::IO::InputSpec> specs_;
  };

}  // namespace Core::Conditions


FOUR_C_NAMESPACE_CLOSE

#endif
