/**
 * Created by Chang on 2/16/18.
 */
//public class deserialize {
//
//    class TreeNode{
//        int val;
//
//        public TreeNode(){
//            val = 0;
//        };
//
//        public int getCildren(){
//            return this.val;
//        }
//
//    }
//    public void serialize(TreeNode root, StringBuilder out)  {
//        if  (root != null)  {
//            out.append('(');
//            out.append(root.val);
//            for  (TreeNode child : root.getChildren())  {
//                serialize(child, out);
//            }
//            out.append(')');
//        }
//    }
//
//    TreeNode deserialize(StringBuilder in) {
//        if  (0  <  in.length()) {
//            if (in.charAt(0)  ==  '(') {
//                in.deleteCharAt(0);
//                TreeNode node = new TreeNode(in.charAt(0) - '0');
//                in.deleteCharAt(0);
//                while  (in.charAt(0)  !=  ')') {
//                    node.add(deserialize(in));
//                }
//                in.deleteCharAt(0);
//                return node;
//            }
//        }
//        return null;
//    }
//
//
//}
