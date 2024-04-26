/*----------------------------------------------------------------------*/
/*! \file

 \brief provides a class with search tree

 \level 1

 */
#include "4C_discretization_geometry_searchtree.hpp"

#include "4C_discretization_geometry_intersection_service.hpp"
#include "4C_discretization_geometry_intersection_service_templates.hpp"
#include "4C_discretization_geometry_position_array.hpp"
#include "4C_discretization_geometry_searchtree_service.hpp"
#include "4C_io_gmsh.hpp"

#include <Teuchos_TimeMonitor.hpp>

#include <fstream>

FOUR_C_NAMESPACE_OPEN


CORE::GEO::SearchTree::SearchTree(const int max_depth) : max_depth_(max_depth), tree_root_(nullptr)
{
}

/*----------------------------------------------------------------------*
 | initialize or rebuild tree with possibly new              u.may 07/08|
 | discretization, elements are sorted according to the given map       |
 *----------------------------------------------------------------------*/
void CORE::GEO::SearchTree::initializeTree(const CORE::LINALG::Matrix<3, 2>& nodeBox,
    const std::map<int, std::set<int>>& elementsByLabel, const TreeType treetype)
{
  tree_root_ = Teuchos::null;
  tree_root_ = Teuchos::rcp(new TreeNode(nullptr, max_depth_, nodeBox, treetype));

  // insert element map into tree root node
  if (elementsByLabel.size() > 0) tree_root_->setElementList(elementsByLabel);

  return;
}

/*----------------------------------------------------------------------*
 | initialize or rebuild tree with possibly new              u.may 07/08|
 | discretization, elements are taken unsorted from discretization      |
 *----------------------------------------------------------------------*/
void CORE::GEO::SearchTree::initializeTree(const CORE::LINALG::Matrix<3, 2>& nodeBox,
    const DRT::Discretization& dis, const TreeType treetype)
{
  tree_root_ = Teuchos::null;
  tree_root_ = Teuchos::rcp(new TreeNode(nullptr, max_depth_, nodeBox, treetype));

  // inserts all elements in a map with key -1 and global id
  for (int i = 0; i < dis.NumMyColElements(); i++)
    tree_root_->insertElement(-1, dis.lColElement(i)->Id());
}

/*----------------------------------------------------------------------*
 | initialize or rebuild tree with possibly new              u.may 07/08|
 | discretization, elements are taken unsorted from discretization      |
 *----------------------------------------------------------------------*/
void CORE::GEO::SearchTree::initializeTree(
    const CORE::LINALG::Matrix<3, 2>& nodeBox, const TreeType treetype)
{
  tree_root_ = Teuchos::null;
  tree_root_ = Teuchos::rcp(new TreeNode(nullptr, max_depth_, nodeBox, treetype));
}

void CORE::GEO::SearchTree::insertElement(const int eid)
{
  // inserts all elements in a map with key -1 and global id
  tree_root_->insertElement(-1, eid);
}

/*----------------------------------------------------------------------*
 | initialization of searchtree for SlipAle                 u.may 09/09 |
 *----------------------------------------------------------------------*/
void CORE::GEO::SearchTree::initializeTreeSlideALE(const CORE::LINALG::Matrix<3, 2>& nodeBox,
    std::map<int, Teuchos::RCP<DRT::Element>>& elements, const TreeType treetype)
{
  tree_root_ = Teuchos::null;
  tree_root_ = Teuchos::rcp(new TreeNode(nullptr, max_depth_, nodeBox, treetype));

  std::map<int, Teuchos::RCP<DRT::Element>>::const_iterator elemiter;
  for (elemiter = elements.begin(); elemiter != elements.end(); ++elemiter)
  {
    tree_root_->insertElement(-1, elemiter->first);
  }
}

/*----------------------------------------------------------------------*
 | returns nodes in the radius of a given point              u.may 07/08|
 *----------------------------------------------------------------------*/
std::map<int, std::set<int>> CORE::GEO::SearchTree::searchElementsInRadius(
    const DRT::Discretization& dis,
    const std::map<int, CORE::LINALG::Matrix<3, 1>>& currentpositions,
    const CORE::LINALG::Matrix<3, 1>& point, const double radius, const int label)
{
  TEUCHOS_FUNC_TIME_MONITOR("SearchTree - queryTime");

  std::map<int, std::set<int>> nodeset;

  if (tree_root_ == Teuchos::null) FOUR_C_THROW("tree is not yet initialized !!!");

  if (!(tree_root_->getElementList().empty()))
    nodeset = tree_root_->searchElementsInRadius(dis, currentpositions, point, radius, label);
  else
    FOUR_C_THROW("element list is empty");

  return nodeset;
}


/*------------------------------------------------------------------------------------------------*
 * build the static search tree for the collision detection                           wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void CORE::GEO::SearchTree::buildStaticSearchTree(
    const std::map<int, CORE::LINALG::Matrix<3, 2>>& currentBVs)
{
  TEUCHOS_FUNC_TIME_MONITOR("SearchTree - buildStaticSearchTree");

  if (tree_root_ == Teuchos::null) FOUR_C_THROW("tree is not yet initialized !!!");

  if (!tree_root_->getElementList().empty())
  {
    tree_root_->buildStaticSearchTree(currentBVs);
  }
  else
    FOUR_C_THROW("element list is empty");

  return;
}

/*------------------------------------------------------------------------------------------------*
 * build the static search tree for the collision detection                           wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void CORE::GEO::SearchTree::buildStaticSearchTree(
    const std::map<int, CORE::LINALG::Matrix<9, 2>>& currentBVs)
{
  TEUCHOS_FUNC_TIME_MONITOR("SearchTree - buildStaticSearchTree");

  if (tree_root_ == Teuchos::null) FOUR_C_THROW("tree is not yet initialized !!!");

  if (!tree_root_->getElementList().empty())
  {
    tree_root_->buildStaticSearchTree(currentBVs);
  }
  else
    FOUR_C_THROW("element list is empty");

  return;
}

/*------------------------------------------------------------------------------------------------*
 * detects collisions between query bounding volume and the bounding volumes of the elements      *
 * in the search tree                                                                 wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void CORE::GEO::SearchTree::searchCollisions(
    const std::map<int, CORE::LINALG::Matrix<3, 2>>& currentBVs,
    const CORE::LINALG::Matrix<3, 2>& queryBV, const int label, std::set<int>& collisions)
{
  TEUCHOS_FUNC_TIME_MONITOR("SearchTree - queryTime");

  if (tree_root_ == Teuchos::null) FOUR_C_THROW("tree is not yet initialized !!!");

  if (!tree_root_->getElementList().empty())
  {
    tree_root_->searchCollisions(currentBVs, queryBV, label, collisions);
  }
  else
    FOUR_C_THROW("element list is empty");

  return;
}

/*----------------------------------------------------------------------*
 | returns contact elements for a given element              u.may 02/09|
 | for multibody contact                                                |
 *----------------------------------------------------------------------*/
void CORE::GEO::SearchTree::searchCollisions(
    const std::map<int, CORE::LINALG::Matrix<9, 2>>& currentKDOPs,
    const CORE::LINALG::Matrix<9, 2>& queryKDOP, const int label, std::set<int>& contactEleIds)
{
  TEUCHOS_FUNC_TIME_MONITOR("SearchTree - queryTime");

  if (tree_root_ == Teuchos::null) FOUR_C_THROW("tree is not yet initialized !!!");

  if (!tree_root_->getElementList().empty())
    tree_root_->searchCollisions(currentKDOPs, queryKDOP, label, contactEleIds);
  else
    return;

  return;
}

/*----------------------------------------------------------------------*
 | c-tor TreeNode                                            u.may 07/08|
 *----------------------------------------------------------------------*/
CORE::GEO::SearchTree::TreeNode::TreeNode(const TreeNode* const parent, const int depth,
    const CORE::LINALG::Matrix<3, 2>& nodeBox, const TreeType treeType)
    : parent_(parent),
      treedepth_(depth),
      tree_node_type_(LEAF_NODE),
      tree_type_(treeType),
      label_(-1),
      node_box_(nodeBox),
      x_plane_coordinate_((node_box_(0, 0) + 0.5 * (node_box_(0, 1) - node_box_(0, 0)))),
      y_plane_coordinate_((node_box_(1, 0) + 0.5 * (node_box_(1, 1) - node_box_(1, 0)))),
      z_plane_coordinate_((node_box_(2, 0) + 0.5 * (node_box_(2, 1) - node_box_(2, 0))))
{
  children_.assign(getNumChildren(), Teuchos::null);
}


/*====================================================================*/
/* octtree */
/*--------------------------------------------------------------------*/
/* numbering of children
 * child 0:
 * child 1:
 * child 2:
 * child 3:
 * child 4:
 * child 5:
 * child 6:
 * child 7:
 *
 *                      z
 *                      |
 *             4========|================6
 *           //|        |               /||
 *          // |        |              //||
 *         //  |        |             // ||
 *        //   |        |            //  ||
 *       //    |        |           //   ||
 *      //     |        |          //    ||
 *     //      |        |         //     ||
 *     5=========================7       ||
 *    ||       |        |        ||      ||
 *    ||       |        o--------||---------y
 *    ||       |       /         ||      ||
 *    ||       0------/----------||------2
 *    ||      /      /           ||     //
 *    ||     /      /            ||    //
 *    ||    /      /             ||   //
 *    ||   /      /              ||  //
 *    ||  /      /               || //
 *    || /      x                ||//
 *    ||/                        ||/
 *     1=========================3
 *
 */
/*====================================================================*/

void CORE::GEO::SearchTree::TreeNode::setElementList(
    const std::map<int, std::set<int>>& elementsByLabel)
{
  element_list_ = elementsByLabel;
}

void CORE::GEO::SearchTree::TreeNode::setNearestObject(
    const CORE::GEO::NearestObject& nearestObject)
{
  nearest_object_ = nearestObject;
}

int CORE::GEO::SearchTree::TreeNode::getNumChildren() const
{
  if (tree_type_ == OCTTREE)
    return 8;
  else if (tree_type_ == QUADTREE)
    return 4;
  else
    FOUR_C_THROW("treetype does not exist");

  return -1;
}

Teuchos::RCP<CORE::GEO::SearchTree::TreeNode> CORE::GEO::SearchTree::TreeNode::getChild(
    const int index) const
{
  return children_[index];
}

CORE::LINALG::Matrix<3, 2> CORE::GEO::SearchTree::TreeNode::getChildNodeBox(const int index) const
{
  CORE::LINALG::Matrix<3, 2> childNodeBox;

  // determine x-coordinates
  if ((index == 1) || (index == 3) || (index == 5) || (index == 7))
  {
    childNodeBox(0, 0) = x_plane_coordinate_;
    childNodeBox(0, 1) = node_box_(0, 1);
  }
  else
  {
    childNodeBox(0, 0) = node_box_(0, 0);
    childNodeBox(0, 1) = x_plane_coordinate_;
  }

  // determine y-coordinates
  if ((index == 2) || (index == 3) || (index == 6) || (index == 7))
  {
    childNodeBox(1, 0) = y_plane_coordinate_;
    childNodeBox(1, 1) = node_box_(1, 1);
  }
  else
  {
    childNodeBox(1, 0) = node_box_(1, 0);
    childNodeBox(1, 1) = y_plane_coordinate_;
  }

  // determine z coordinates
  if (index > 3)
  {
    childNodeBox(2, 0) = z_plane_coordinate_;
    childNodeBox(2, 1) = node_box_(2, 1);
  }
  else
  {
    if (tree_type_ == OCTTREE)
    {
      childNodeBox(2, 0) = node_box_(2, 0);
      childNodeBox(2, 1) = z_plane_coordinate_;
    }
    else
    {
      childNodeBox(2, 0) = 0.0;
      childNodeBox(2, 1) = 0.0;
    }
  }
  return childNodeBox;
}

void CORE::GEO::SearchTree::TreeNode::getChildNodeBox(
    const int index, CORE::LINALG::Matrix<3, 2>& childNodeBox) const
{
  childNodeBox.Clear();

  // determine x-coordinates
  if ((index == 1) || (index == 3) || (index == 5) || (index == 7))
  {
    childNodeBox(0, 0) = x_plane_coordinate_;
    childNodeBox(0, 1) = node_box_(0, 1);
  }
  else
  {
    childNodeBox(0, 0) = node_box_(0, 0);
    childNodeBox(0, 1) = x_plane_coordinate_;
  }

  // determine y-coordinates
  if ((index == 2) || (index == 3) || (index == 6) || (index == 7))
  {
    childNodeBox(1, 0) = y_plane_coordinate_;
    childNodeBox(1, 1) = node_box_(1, 1);
  }
  else
  {
    childNodeBox(1, 0) = node_box_(1, 0);
    childNodeBox(1, 1) = y_plane_coordinate_;
  }

  // determine z coordinates
  if (index > 3)
  {
    childNodeBox(2, 0) = z_plane_coordinate_;
    childNodeBox(2, 1) = node_box_(2, 1);
  }
  else
  {
    if (tree_type_ == OCTTREE)
    {
      childNodeBox(2, 0) = node_box_(2, 0);
      childNodeBox(2, 1) = z_plane_coordinate_;
    }
    else
    {
      childNodeBox(2, 0) = 0.0;
      childNodeBox(2, 1) = 0.0;
    }
  }
  return;
}


void CORE::GEO::SearchTree::TreeNode::insertElement(const int label, const int eleId)
{
  element_list_[label].insert(eleId);
  return;
}

void CORE::GEO::SearchTree::TreeNode::createChildren(const DRT::Discretization& dis,
    const std::map<int, CORE::LINALG::Matrix<3, 1>>& currentpositions)
{
  // create empty children
  for (int index = 0; index < getNumChildren(); index++)
    children_[index] =
        Teuchos::rcp(new TreeNode(this, (treedepth_ - 1), getChildNodeBox(index), tree_type_));

  // insert elements into child node
  for (std::map<int, std::set<int>>::const_iterator labelIter = element_list_.begin();
       labelIter != element_list_.end(); labelIter++)
    for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
         eleIter != (labelIter->second).end(); eleIter++)
    {
      std::vector<int> elementClassification =
          classifyElement(dis.gElement(*eleIter), currentpositions);
      for (unsigned int count = 0; count < elementClassification.size(); count++)
      {
        children_[elementClassification[count]]->insertElement(labelIter->first, *eleIter);
      }
    }

  // this node becomes an inner tree node
  tree_node_type_ = INNER_NODE;
}


void CORE::GEO::SearchTree::TreeNode::createChildren(
    const std::map<int, CORE::LINALG::Matrix<3, 2>>& currentXAABBs)
{
  // create empty children
  CORE::LINALG::Matrix<3, 2> childNodeBox;
  for (int index = 0; index < getNumChildren(); index++)
  {
    getChildNodeBox(index, childNodeBox);
    children_[index] = Teuchos::rcp(new TreeNode(this, (treedepth_ - 1), childNodeBox, tree_type_));
  }
  std::vector<int> elementClassification;
  // insert elements into child node
  for (std::map<int, std::set<int>>::const_iterator labelIter = element_list_.begin();
       labelIter != element_list_.end(); labelIter++)
  {
    for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
         eleIter != (labelIter->second).end(); eleIter++)
    {
      classifyXAABB(currentXAABBs.find(*eleIter)->second, elementClassification);
      for (unsigned int count = 0; count < elementClassification.size(); count++)
        children_[elementClassification[count]]->insertElement(labelIter->first, *eleIter);
    }
  }
  // this node becomes an inner tree node
  tree_node_type_ = INNER_NODE;
  return;
}

/*----------------------------------------------------------------------*
 | create children using KDOPS                             u.may   02/09|
 *----------------------------------------------------------------------*/
void CORE::GEO::SearchTree::TreeNode::createChildren(
    const std::map<int, CORE::LINALG::Matrix<9, 2>>& currentKDOPs)
{
  // create empty children
  static CORE::LINALG::Matrix<3, 2> childNodeBox;
  for (int index = 0; index < getNumChildren(); index++)
  {
    getChildNodeBox(index, childNodeBox);
    children_[index] = Teuchos::rcp(new TreeNode(this, (treedepth_ - 1), childNodeBox, tree_type_));
  }

  static std::vector<int> elementClassification;
  // insert elements into child node
  for (std::map<int, std::set<int>>::const_iterator labelIter = element_list_.begin();
       labelIter != element_list_.end(); labelIter++)
  {
    for (std::set<int>::const_iterator eleIter = (labelIter->second).begin();
         eleIter != (labelIter->second).end(); eleIter++)
    {
      classifyKDOP(currentKDOPs.find(*eleIter)->second, elementClassification);
      for (unsigned int count = 0; count < elementClassification.size(); count++)
        children_[elementClassification[count]]->insertElement(labelIter->first, *eleIter);
    }
  }
  // this node becomes an inner tree node
  tree_node_type_ = INNER_NODE;
}


/*----------------------------------------------------------------------*
 | classifiy AABB in node                                  u.may   07/08|
 *----------------------------------------------------------------------*/
std::vector<int> CORE::GEO::SearchTree::TreeNode::classifyXAABB(
    const CORE::LINALG::Matrix<3, 2>& AABB) const
{
  // collect all children the XAABB is lying in
  // use tolerances such that it is ensured that no child is left behind :-)
  // XAABB s which are lying between plane-tol and plane + tol are collected in each of the two
  // chlidren

  std::vector<int> octants;

  // check max_x greater than x-plane
  if (AABB(0, 1) > (x_plane_coordinate_ - CORE::GEO::TOL7))
  {
    // check max_y greater than y-plane
    if (AABB(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(7);

        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(3);
      }
      else if (tree_type_ == QUADTREE)
        octants.push_back(3);
    }

    // check min_y less than y-plane
    if (AABB(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(5);

        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(1);
      }
      else if (tree_type_ == QUADTREE)
        octants.push_back(1);
    }
  }

  // check min_x less than x-plane
  if (AABB(0, 0) < (x_plane_coordinate_ + CORE::GEO::TOL7))
  {
    // check min_y less than y-plane
    if (AABB(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(0);

        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(4);
      }

      else if (tree_type_ == QUADTREE)
        octants.push_back(0);
    }

    // check max_y greater than y-plane
    if (AABB(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(6);

        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(2);
      }

      else if (tree_type_ == QUADTREE)
        octants.push_back(2);
    }
  }
  return octants;
}

/*----------------------------------------------------------------------*
 | classifiy AABB in node                                  u.may   07/08|
 *----------------------------------------------------------------------*/
void CORE::GEO::SearchTree::TreeNode::classifyXAABB(
    const CORE::LINALG::Matrix<3, 2>& AABB, std::vector<int>& octants) const
{
  // collect all children the XAABB is lying in
  // use tolerances such that it is ensured that no child is left behind :-)
  // XAABB s which are lying between plane-tol and plane + tol are collected in each of the two
  // chlidren

  octants.clear();
  octants.reserve(8);

  // check max_x greater than x-plane
  if (AABB(0, 1) > (x_plane_coordinate_ - CORE::GEO::TOL7))
  {
    // check max_y greater than y-plane
    if (AABB(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(7);

        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(3);
      }
      else if (tree_type_ == QUADTREE)
        octants.push_back(3);
    }

    // check min_y less than y-plane
    if (AABB(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(5);

        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(1);
      }
      else if (tree_type_ == QUADTREE)
        octants.push_back(1);
    }
  }

  // check min_x less than x-plane
  if (AABB(0, 0) < (x_plane_coordinate_ + CORE::GEO::TOL7))
  {
    // check min_y less than y-plane
    if (AABB(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(0);

        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(4);
      }
      else if (tree_type_ == QUADTREE)
        octants.push_back(0);
    }

    // check max_y greater than y-plane
    if (AABB(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(6);

        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(2);
      }
      else if (tree_type_ == QUADTREE)
        octants.push_back(2);
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 | classifiy AABB in node                                  u.may   02/09|
 *----------------------------------------------------------------------*/
void CORE::GEO::SearchTree::TreeNode::classifyKDOP(
    const CORE::LINALG::Matrix<9, 2>& KDOP, std::vector<int>& octants) const
{
  // collect all children the XAABB is lying in
  // use tolerances such that it is ensured that no child is left behind :-)
  // XAABB s which are lying between plane-tol and plane + tol are collected in each of the two
  // chlidren

  octants.clear();
  octants.reserve(8);

  // check max_x greater than x-plane
  if (KDOP(0, 1) > (x_plane_coordinate_ - CORE::GEO::TOL7))
  {
    // check max_y greater than y-plane
    if (KDOP(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      // check max_z greater than z-plane
      if (KDOP(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(7);

      // check min_z less than z-plane
      if (KDOP(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(3);
    }

    // check min_y less than y-plane
    if (KDOP(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      // check max_z greater than z-plane
      if (KDOP(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(5);

      // check min_z less than z-plane
      if (KDOP(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(1);
    }
  }

  // check min_x less than x-plane
  if (KDOP(0, 0) < (x_plane_coordinate_ + CORE::GEO::TOL7))
  {
    // check min_y less than y-plane
    if (KDOP(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      // check min_z less than z-plane
      if (KDOP(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(0);

      // check max_z greater than z-plane
      if (KDOP(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(4);
    }

    // check max_y greater than y-plane
    if (KDOP(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      // check max_z greater than z-plane
      if (KDOP(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7)) octants.push_back(6);

      // check min_z less than z-plane
      if (KDOP(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7)) octants.push_back(2);
    }
  }
  return;
}

/*----------------------------------------------------------------------*
 | classifiy AABB in node                                  u.may   07/08|
 *----------------------------------------------------------------------*/
bool CORE::GEO::SearchTree::TreeNode::classifyXAABB(
    int& index, const CORE::LINALG::Matrix<3, 2>& AABB) const
{
  // collect all children the XAABB is lying in
  // use tolerances such that it is ensured that no child is left behind :-)
  // XAABB s which are lying between plane-tol and plane + tol are collected in each of the two
  // chlidren

  bool oneIndex = true;
  index = -1;

  // check max_x greater than x-plane
  if (AABB(0, 1) > (x_plane_coordinate_ - CORE::GEO::TOL7))
  {
    // check max_y greater than y-plane
    if (AABB(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 7;
          else
          {
            index = -1;
            return false;
          }
        }
        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 3;
          else
          {
            index = -1;
            return false;
          }
        }
      }
      else if (tree_type_ == QUADTREE)
      {
        if (index == -1)
          index = 3;
        else
        {
          index = -1;
          return false;
        }
      }
    }

    // check min_y less than y-plane
    if (AABB(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 5;
          else
          {
            index = -1;
            return false;
          }
        }

        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 1;
          else
          {
            index = -1;
            return false;
          }
        }
      }
      else if (tree_type_ == QUADTREE)
      {
        if (index == -1)
          index = 1;
        else
        {
          index = -1;
          return false;
        }
      }
    }
  }

  // check min_x less than x-plane
  if (AABB(0, 0) < (x_plane_coordinate_ + CORE::GEO::TOL7))
  {
    // check min_y less than y-plane
    if (AABB(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 0;
          else
          {
            index = -1;
            return false;
          }
        }
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 4;
          else
          {
            index = -1;
            return false;
          }
        }
      }
      else if (tree_type_ == QUADTREE)
      {
        if (index == -1)
          index = 0;
        else
        {
          index = -1;
          return false;
        }
      }
    }

    // check max_y greater than y-plane
    if (AABB(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (AABB(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 6;
          else
          {
            index = -1;
            return false;
          }
        }

        // check min_z less than z-plane
        if (AABB(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 2;
          else
          {
            index = -1;
            return false;
          }
        }
      }
      else if (tree_type_ == QUADTREE)
      {
        if (index == -1)
          index = 2;
        else
        {
          index = -1;
          return false;
        }
      }
    }
  }
  return oneIndex;
}

/*----------------------------------------------------------------------*
 | classifiy KDOP in node                                  u.may   02/09|
 *----------------------------------------------------------------------*/
bool CORE::GEO::SearchTree::TreeNode::classifyKDOP(
    int& index, const CORE::LINALG::Matrix<9, 2>& KDOP) const
{
  // collect all children the XAABB is lying in
  // use tolerances such that it is ensured that no child is left behind :-)
  // XAABB s which are lying between plane-tol and plane + tol are collected in each of the two
  // chlidren
  bool oneIndex = true;
  index = -1;

  // check max_x greater than x-plane
  if (KDOP(0, 1) > (x_plane_coordinate_ - CORE::GEO::TOL7))
  {
    // check max_y greater than y-plane
    if (KDOP(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (KDOP(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 7;
          else
          {
            index = -1;
            return false;
          }
        }
        // check min_z less than z-plane
        if (KDOP(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 3;
          else
          {
            index = -1;
            return false;
          }
        }
      }
      else if (tree_type_ == QUADTREE)
      {
        if (index == -1)
          index = 3;
        else
        {
          index = -1;
          return false;
        }
      }
    }

    // check min_y less than y-plane
    if (KDOP(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (KDOP(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 5;
          else
          {
            index = -1;
            return false;
          }
        }

        // check min_z less than z-plane
        if (KDOP(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 1;
          else
          {
            index = -1;
            return false;
          }
        }
      }
      else if (tree_type_ == QUADTREE)
      {
        if (index == -1)
          index = 1;
        else
        {
          index = -1;
          return false;
        }
      }
    }
  }

  // check min_x less than x-plane
  if (KDOP(0, 0) < (x_plane_coordinate_ + CORE::GEO::TOL7))
  {
    // check min_y less than y-plane
    if (KDOP(1, 0) < (y_plane_coordinate_ + CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check min_z less than z-plane
        if (KDOP(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 0;
          else
          {
            index = -1;
            return false;
          }
        }
        // check max_z greater than z-plane
        if (KDOP(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 4;
          else
          {
            index = -1;
            return false;
          }
        }
      }
      else if (tree_type_ == QUADTREE)
      {
        if (index == -1)
          index = 0;
        else
        {
          index = -1;
          return false;
        }
      }
    }

    // check max_y greater than y-plane
    if (KDOP(1, 1) > (y_plane_coordinate_ - CORE::GEO::TOL7))
    {
      if (tree_type_ == OCTTREE)
      {
        // check max_z greater than z-plane
        if (KDOP(2, 1) > (z_plane_coordinate_ - CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 6;
          else
          {
            index = -1;
            return false;
          }
        }

        // check min_z less than z-plane
        if (KDOP(2, 0) < (z_plane_coordinate_ + CORE::GEO::TOL7))
        {
          if (index == -1)
            index = 2;
          else
          {
            index = -1;
            return false;
          }
        }
      }
      else if (tree_type_ == QUADTREE)
      {
        if (index == -1)
          index = 2;
        else
        {
          index = -1;
          return false;
        }
      }
    }
  }
  return oneIndex;
}

/*----------------------------------------------------------------------*
 | classifiy element in node                               peder   07/08|
 *----------------------------------------------------------------------*/
std::vector<int> CORE::GEO::SearchTree::TreeNode::classifyElement(const DRT::Element* element,
    const std::map<int, CORE::LINALG::Matrix<3, 1>>& currentpositions) const
{
  const CORE::LINALG::SerialDenseMatrix xyze(
      CORE::GEO::getCurrentNodalPositions(element, currentpositions));
  CORE::GEO::EleGeoType eleGeoType(CORE::GEO::HIGHERORDER);
  CORE::GEO::checkRoughGeoType(element, xyze, eleGeoType);
  const CORE::LINALG::Matrix<3, 2> elemXAABB(
      CORE::GEO::computeFastXAABB(element->Shape(), xyze, eleGeoType));
  return classifyXAABB(elemXAABB);
}

/*----------------------------------------------------------------------*
 | classifiy element in tree node                           u.may   07/08|
 *----------------------------------------------------------------------*/
std::vector<int> CORE::GEO::SearchTree::TreeNode::classifyElement(
    const Teuchos::RCP<DRT::Element> element,
    const std::map<int, CORE::LINALG::Matrix<3, 1>>& currentpositions) const
{
  const CORE::LINALG::SerialDenseMatrix xyze(
      CORE::GEO::getCurrentNodalPositions(element, currentpositions));
  CORE::GEO::EleGeoType eleGeoType(CORE::GEO::HIGHERORDER);
  CORE::GEO::checkRoughGeoType(element, xyze, eleGeoType);
  const CORE::LINALG::Matrix<3, 2> elemXAABB(
      CORE::GEO::computeFastXAABB(element->Shape(), xyze, eleGeoType));
  return classifyXAABB(elemXAABB);
}

/*----------------------------------------------------------------------*
 | classifiy element in tree node                          u.may   07/08|
 *----------------------------------------------------------------------*/
std::vector<int> CORE::GEO::SearchTree::TreeNode::classifyElement(
    const DRT::Element* element, const CORE::LINALG::SerialDenseMatrix& xyze_element) const
{
  CORE::GEO::EleGeoType eleGeoType(CORE::GEO::HIGHERORDER);
  CORE::GEO::checkRoughGeoType(element, xyze_element, eleGeoType);
  const CORE::LINALG::Matrix<3, 2> elemXAABB(
      CORE::GEO::computeFastXAABB(element->Shape(), xyze_element, eleGeoType));
  return classifyXAABB(elemXAABB);
}

/*----------------------------------------------------------------------*
 | classifiy element in node                               u.may   08/08|
 *----------------------------------------------------------------------*/
std::vector<int> CORE::GEO::SearchTree::TreeNode::classifyRadius(
    const double radius, const CORE::LINALG::Matrix<3, 1>& point) const
{
  /*coordinates of axis-aligned boundary box around point, which stretechs over
   * length radius in each coordinate direction*/
  CORE::LINALG::Matrix<3, 2> radiusXAABB;

  for (int dim = 0; dim < 3; dim++)
  {
    radiusXAABB(dim, 0) = (point(dim) - radius) - CORE::GEO::TOL7;
    radiusXAABB(dim, 1) = (point(dim) + radius) + CORE::GEO::TOL7;
  }

  if (tree_type_ == QUADTREE)
  {
    radiusXAABB(2, 0) = 0.0;
    radiusXAABB(2, 1) = 0.0;
  }

  // return indices of childs which overlap with given axis-aligned boundary box of edge length
  // 2*radius around point
  return classifyXAABB(radiusXAABB);
}

/*----------------------------------------------------------------------*
 | returns nodes in the radius of a given point              u.may 08/08|
 *----------------------------------------------------------------------*/
std::map<int, std::set<int>> CORE::GEO::SearchTree::TreeNode::searchElementsInRadius(
    const DRT::Discretization& dis,
    const std::map<int, CORE::LINALG::Matrix<3, 1>>& currentpositions,
    const CORE::LINALG::Matrix<3, 1>& point, const double radius, const int label)
{
  std::map<int, std::set<int>> eleMap;

  switch (tree_node_type_)
  {
    case INNER_NODE:
    {
      const std::vector<int> childindex = classifyRadius(radius, point);
      if (childindex.size() < 1)
        FOUR_C_THROW("no child found\n");
      else if (childindex.size() == 1)  // child node found which encloses AABB so step down
        return children_[childindex[0]]->searchElementsInRadius(
            dis, currentpositions, point, radius, label);
      else
        return CORE::GEO::getElementsInRadius(
            dis, currentpositions, point, radius, label, element_list_);
      break;
    }
    case LEAF_NODE:
    {
      if (element_list_.empty()) return eleMap;

      // max depth reached, counts reverse
      if (treedepth_ <= 0 ||
          (element_list_.size() == 1 && (element_list_.begin()->second).size() == 1))
        return CORE::GEO::getElementsInRadius(
            dis, currentpositions, point, radius, label, element_list_);

      // dynamically grow tree otherwise, create children and set label for empty children
      // search in appropriate child node
      const std::vector<int> childindex = classifyRadius(radius, point);
      if (childindex.size() < 1)
        FOUR_C_THROW("no child found\n");
      else if (childindex.size() == 1)  // child node found which encloses AABB so refine further
      {
        createChildren(dis, currentpositions);
        return children_[childindex[0]]->searchElementsInRadius(
            dis, currentpositions, point, radius, label);
      }
      else
        // AABB does not fit into a single child node box anymore so don t refine any further
        return CORE::GEO::getElementsInRadius(
            dis, currentpositions, point, radius, label, element_list_);
      break;
    }
    default:
      FOUR_C_THROW("should not get here\n");
  }
  return eleMap;
}

/*------------------------------------------------------------------------------------------------*
 * build the static search tree for the collision detection                           wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void CORE::GEO::SearchTree::TreeNode::buildStaticSearchTree(
    const std::map<int, CORE::LINALG::Matrix<3, 2>>& currentBVs)
{
  if (element_list_.empty()) return;

  if (treedepth_ > 0 and (element_list_.begin()->second).size() >
                             1)  // ************************* > 8 could be interesting
  {
    createChildren(currentBVs);
    for (int count = 0; count < getNumChildren(); count++)
      children_[count]->buildStaticSearchTree(currentBVs);
    return;
  }
  return;
}

/*------------------------------------------------------------------------------------------------*
 * build the static search tree for the collision detection                           wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void CORE::GEO::SearchTree::TreeNode::buildStaticSearchTree(
    const std::map<int, CORE::LINALG::Matrix<9, 2>>& currentBVs)
{
  if (element_list_.empty()) return;

  if (treedepth_ > 0 and (element_list_.begin()->second).size() >
                             1)  // ************************* > 8 could be interesting
  {
    createChildren(currentBVs);
    for (int count = 0; count < getNumChildren(); count++)
      children_[count]->buildStaticSearchTree(currentBVs);
    return;
  }
  return;
}

/*------------------------------------------------------------------------------------------------*
 * search collisions at the leaf nodes                                                wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void CORE::GEO::SearchTree::TreeNode::searchCollisions(
    const std::map<int, CORE::LINALG::Matrix<3, 2>>& currentBVs,
    const CORE::LINALG::Matrix<3, 2>& queryBV, const int label, std::set<int>& collisions)
{
  switch (tree_node_type_)
  {
    case INNER_NODE:
    {
      std::vector<int> elementClassification;
      classifyXAABB(queryBV, elementClassification);
      for (unsigned int count = 0; count < elementClassification.size(); count++)
        children_[elementClassification[count]]->searchCollisions(
            currentBVs, queryBV, label, collisions);
      return;
      break;
    }
    case LEAF_NODE:
    {
      if (element_list_.empty()) return;

      // max depth reached, counts reverse
      if (treedepth_ <= 0 || (element_list_.begin()->second).size() == 1)
      {
        CORE::GEO::searchCollisions(currentBVs, queryBV, label, element_list_, collisions);
        return;
      }
      // dynamically grow tree otherwise, create children and set label for empty children
      // search in apropriate child node
      createChildren(currentBVs);
      std::vector<int> elementClassification;
      classifyXAABB(queryBV, elementClassification);
      for (unsigned int count = 0; count < elementClassification.size(); count++)
        children_[elementClassification[count]]->searchCollisions(
            currentBVs, queryBV, label, collisions);
      return;
      break;
    }
    default:
      FOUR_C_THROW("should not get here\n");
  }
  return;
}

/*------------------------------------------------------------------------------------------------*
 * search collisions at the leaf nodes                                                wirtz 08/14 *
 *------------------------------------------------------------------------------------------------*/
void CORE::GEO::SearchTree::TreeNode::searchCollisions(
    const std::map<int, CORE::LINALG::Matrix<9, 2>>& currentBVs,
    const CORE::LINALG::Matrix<9, 2>& queryBV, const int label, std::set<int>& collisions)
{
  switch (tree_node_type_)
  {
    case INNER_NODE:
    {
      std::vector<int> elementClassification;
      classifyKDOP(queryBV, elementClassification);
      for (unsigned int count = 0; count < elementClassification.size(); count++)
        children_[elementClassification[count]]->searchCollisions(
            currentBVs, queryBV, label, collisions);
      return;
      break;
    }
    case LEAF_NODE:
    {
      if (element_list_.empty()) return;

      // max depth reached, counts reverse
      if (treedepth_ <= 0 || (element_list_.begin()->second).size() == 1)
      {
        CORE::GEO::searchCollisions(currentBVs, queryBV, label, element_list_, collisions);
        return;
      }
      // dynamically grow tree otherwise, create children and set label for empty children
      // search in apropriate child node
      createChildren(currentBVs);
      std::vector<int> elementClassification;
      classifyKDOP(queryBV, elementClassification);
      for (unsigned int count = 0; count < elementClassification.size(); count++)
        children_[elementClassification[count]]->searchCollisions(
            currentBVs, queryBV, label, collisions);
      return;
      break;
    }
    default:
      FOUR_C_THROW("should not get here\n");
  }
  return;
}

FOUR_C_NAMESPACE_CLOSE