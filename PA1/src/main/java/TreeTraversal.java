/**
 * Created by Chang on 11/8/17.
 */
import java.util.*;
public class TreeTraversal {


        class TreeNode{
            int val;
            TreeNode left;
            TreeNode right;
            public TreeNode(int val){
                this.val = val;
                this.left = null;
                this.right = null;
            }

        }
        public List<Integer> inorderTraversal(TreeNode root){
            List<Integer> res = new ArrayList<>();
            if(root == null) return res;
            Stack<TreeNode> stack = new Stack();
            TreeNode p = root;
            while(!stack.isEmpty() || p != null){
                if(p != null){
                    stack.push(p);
                    p = p.left;
                }else{
                    TreeNode n = stack.pop();
                    res.add(n.val);
                    p = n.right;
                }
            }
            return res;
        }
}

