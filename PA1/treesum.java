//inorder(mirrors O(1) space),preorder, bsf, dfs, treepath, subtree-sum
public class TreeNode{
	int val;
	TreeNode left;
	TreeNode right;
	TreeNode(int x){
		val = x;
	}
}
public class Solution{
	public List<Integer> inorderTraversal(TreeNode root){
		List<Integer> list = new ArrayList<Integer>();
		if (root == null) return list;
		Stack<TreeNode> stack = new Stack<TreeNode>();
		TreeNode p = root;
		while(!stack.isEmpty()|| p != null){
			if(p != null){
				stack.push(p);
				p = p.left;

			}
			else{
				TreeNode t = stack.pop();
				list.add(t);
				p = t.right;
			}

		}
		return list;
	}
}
public class Solution{
	public boolean isValidBST(TreeNode root){
		ArrayList<Integer> result = new ArrayList<>();
		inorder(root, result);
		for(int i = 1; i<result.size(); i++){
			if(result.get(i)<= result.get(i-1)) return false;
		}
		return true;

	}
	private void inorder(TreeNode root, ArrayList<Integer> result){
		if(root == null) return;
		inorder(root.left, result);
		result.add(root.val);
		inorder(root.right, result);
	}
}

public int Mindepth(TreeNode root){
	if (root == null) return 0;
	int ldepth = Mindepth(root.left);
	int rdepth = MIndepth(root.right);
	if(ldepth == 0){
		return 1+rdepth;
	}
	else if(rdepth == 0){
		return 1+ldepth;
	}
	return (1+ Math.min(rdepth, ldepth));
}
public int Mindepth(TreeNode root){
	if(root == null) return 0;
	int depth = 0;
	if(root.left == null || root.right == null){
		depth = 1+ Math.max(Mindepth(root.left), Mindepth(root.right));
	}
	else{
		depth = 1 + Math.min(Mindepth(root.left), Mindepth(root.right));
	}
	return depth;
}
public int maxDepth(TreeNode root){
	int count = 0;
	if(root == null) return count;
	Queue <TreeNode> stack = new LinkedList<TreeNode>();
	stack.offer(root);
	while(!stack.isEmpty()){
		int size = stack.size();
		whie(size>0){
			TreeNode cur = stack.poll();
			if(cur.left != null) stack.offer(cur.left);
			if(cur.right != null) stack.offer(cur.right);
			size--;
		}
		count++
	}
	return count;

}
putlic ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
	ArrayList<ArrayList<Integer>> res = new ArrayList();
	if (root == null) return res;
	Queue<TreeNode> queue = new LinkedList<TreeNode>();
	queue.offer(root);
	while(!queue.isEmpty()){
		ArrayList<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i< queue.size(); i++){
			TreeNode cur = queue.poll();
			list.add(cur.val);
			if(cur.left != null) queue.offer(cur.left);
			if(cur.right != null) queue.offer(cur.right);

		}
		res.add(list);
	}
	return res;
}

//path sum
public boolean hasPathSum(TreeNode root, int sum){
	if(root == null) return false;
	if(root.left == null && root.right == null && root.val == sum) return true;
	return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);

}

public int PathSum(TreeNode root, in sum){
	if (root == null) return 0;
	int res = dfs(root, sum) + PathSum(root.left, sum) + PathSum(root.right, sum);
	return res;
}
private int dfs(TreeNode root, int sum){
	int res = 0;
	if(root == null) return res;
	if(root.val == sum) res += 1;
	res += dfs(root.left, sum - root.val);
	res += dfs(root.right, sum - root.val);
	return res;
}

public int PathSum(TreeNode root, int sum){
	HashMap<Integer, Integer> map = new HashMap();
	map.put(0,1);
	return helper(root, map, sum, 0);
}
private int helper(TreeNode root, HashMap<Integer,Integer> map, int target, int pre){
	if(root == null) return 0;
	pre += root.val;
	int res = map.getOrDefault(pre-sum, 0);
	map.put(pre, map.getOrDefault(pre, 0) + 1);
	res += helper(root.left, map, sum, pre) + helper(root.right, map, sum, pre);
	map.put(pre, map.get(pre) -1 );
	return res;
}

















