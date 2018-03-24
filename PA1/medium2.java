public List<List<Integer>> combinationSum(int[] candidates, int target) {
	List<List<Integer>> res = new ArrayList<List<Integer>>();
	List<Integer> list = new ArrayList<Integer>();
	Arrays.sort(candidates);
	help(res, candidates, target, list, 0);
	return res;

}
private void help(List<List<Integer>> res, int[] candidates, int target, List<Integer> list, int start){
	if(target ==0){
		res.add(new ArrayList<Integer>(list));
		return;
	}
	for(int i = start; i< candidates.length; i++){
		if(candidates[i]<= target) {
			list.add(candidates[i]);
			help(res, candidates, target-candidates[i], list, i);
			list.remove(list.size() - 1);
		}
}
public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> list = new ArrayList<Integer>();
        Arrays.sort(candidates);
        Help(res, list,candidates, target,0);
        return res;
    }
    private void Help(List<List<Integer>> res, List<Integer> list, int[] candidates, int target,int start){
        if(target < 0) return;
        if(target == 0){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        
        else{
            for(int i = start; i< candidates.length; i++){
              //  if(i> start && candidates[i] == candidates[i-1]) continue;
                list.add(candidates[i]);
                Help(res, list,candidates, target-candidates[i], i);
                list.remove(list.size() - 1);
            }
        }
    }

      public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if(k > n|| n == 0|| k ==0) return res;
        List<Integer> list = new ArrayList<Integer>();
        help(res, n, 1, k, list);
        return res;
        
    }
    private void help( List<List<Integer>> res, int n, int start, int k, List<Integer> list){
        if ( k == 0){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        else{
            for(int i = start; i<=n; i++){
                list.add(i);
                help(res, n, i+1,k-1,list);
                list.remove(list.size() - 1);
            }
        }
    }
        public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if(n == 0 || k == 0) return res;
        List<Integer> list = new ArrayList<Integer>();
        help(res, k, n, 1, list);
        return res;
        
    }
    private void help(List<List<Integer>> res, int k, int sum, int start, List<Integer> list){
        if(k == 0 && sum ==0){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        else{
            for(int i = start; i<=9; i++){
                if(i<= sum) {
                    list.add(i);
                    help(res, k-1, sum-i,i+1, list);
                    list.remove(list.size() - 1);
                }
            }
        }
    }
    public ListNode swapParis(ListNode head){
    	if(head == null|| head.next == null) return head;
    	ListNode dummy = new ListNode(0);
    	dummy.next = head;
    	ListNode cur = dummy;
    	while(cur.next != null && cur.next.next != null){
    		ListNode n1 = cur.next;
    		ListNode n2 = cur.next.next;
    		cur.next = n2
    		n1.next = n2.next;
    		cur.next.next = n1;

    		cur = n1;
    	}
    	return dummy.next;
    }

    class ListNode{
    	int val;
    	ListNode next;
    	ListNode(int x){
    		val = x;
    	}
    }
    public ListNode swap2(ListNode head, int k){
    	ListNode dummy = new ListNode(0);
    	dummy.next = head;
    	ListNode slow = dummy;
    	ListNode fast = dummy;
    	for(int i =0; i<k-1; i++){
    		fast = fast.next;
    		if(fast.next == null) return dummy.next;
    	}
    	ListNode pre1 = fast;
    	fast = fast.next;
    	while(fast.next != null){
    		fast = fast.next;
    		slow = slow.next;
    	}
    	ListNode n1 = pre1.next;
    	ListNode n2 = slow.next.next;
    	pre1.next = slow.next;
    	slow.next.next = n1.next;
    	slow.next = n1;
    	n1.next = n2;
//
    	ListNode n1 = pre1.next;
    	ListNode n2 = pre1.next.next;
    	n1.next = slow.next.next;
    	slow.next.next = n2;
    	pre1.next = slow.next;
    	slow.next = n1;
    	return dummy.next;
  }
  public ListNode swap2(ListNode head, int k){
  	List dummy = new ListNode(0);
  	dummy.next = head;
  	ListNode fast = dummy;
  	for(int i = 0; i<k-1; i++){
  		fast = fast.next;
  		if(fast.next == null) return dummuy.next;// judge if k is larger than the length of linkedlist.
  	}
  	ListNode l1 = fast;
  	fast = fast.next;
  	ListNode slow = dummy;
  	while(fast.next != null){
  		fast = fast.next;
  		slow = slow.next;

  	}
  	ListNode l2 = slow;
  	slow = slow.next;
  	fast = l1.next;
  	if(slow == fast) return dummy.next;// judge if they are the same pointer.
  	ListNode tmp = slow.next;
  	//five pointers, pre1/pre2/fast/slow/tmp= slow.next
  	// only judge on the slow.next pointer as two listnode may be adjacent to each other.
  	l1.next = slow;
  	slow.next = fast.next;
  	l2.next = fast;
  	fast.next = tmp;
  	//
  	l1.next = slow;
  	if(fast.next != slow)
  		slow.next = fast.next;
  	else
  		slow.next = fast;
  	fast.next = tmp;
  	l2.next = fast;
  	
  }
  //
  public boolean canJump(int[] nums) {
     
        int n = nums.length;
        int total = 0;
        for(int i= 0; i<n && i<=total; i++){
            total = Math.max(i+nums[i], total);
            if(total >= n-1) return true;
        }
        
        return false;
}
public int search(int[] nums, int target){
	int left = 0;
	int right = nums.length - 1;
	while(left<= right){
		int mid = left + (right - left)/2;
		if(target == nums[mid]) return mid;
		if(nums[left] <= nums[mid]){
			if(target<nums[mid] && target>=nums[left]) right = mid-1;
			else left = mid + 1;
		}
		else if(nums[mid] < nums[left]){
			if(target>nums[mid] && target <= nums[right]) left = mid + 1;
			else right = mid - 1;
		}
	}
	return -1;
}
public int findMin(int[] nums) {
        if(nums.length ==0 || nums == null) return 0;
        int start = 0;
        int end = nums.length - 1;
        while(start < end){
            int mid = start + (end - start)/2;
            if(nums[mid] > nums[end]){
                start = mid+1;
            }
            else end = mid;
            
        }
        return nums[start];
        
    }
public int findMin(int[] nums) {
        if(nums.length == 0 || nums == null) return 0;
        int start = 0;
        int end = nums.length - 1;
        while(start<end){
            int mid = start +(end - start)/2;
            if(nums[mid]>nums[end]) start = mid + 1;
            else if(nums[mid] < nums[end]) end = mid;
            else if(nums[mid] == nums[end]) end = end -1;
        }
        return nums[end];
        
    }

 public List<Interval> merge(List<Interval> intervals) {
        List<Interval> res = new ArrayList<Interval>();
        if(intervals == null || intervals.size() == 0) return res;
        Collections.sort(intervals, new Comparator<Interval>(){
            public int compare(Interval a, Interval b){
                return a.start != b.start? a.start - b.start : a.end - b.end;
            }
        });
        int start = intervals.get(0).start;
        int end = intervals.get(0).end;
        for(Interval it:intervals){
            if(it.start <= end){
                end = Math.max(it.end, end);
            }else{
                res.add(new Interval(start,end));
                start = it.start;
                end = it.end;
            }
        }
       res.add(new Interval(start,end));
       return res;
        
    }
 public List<Interval> merge(List<Interval> intervals){
	if(intervals == null || intervals.size() <= 1) return intervals;
	Collections.sort(intervals, new Comparator<Interval>(){
		public int compare(Interval a, Interval b){
			return a.start != b.start? a.start - b.start: a.end - b.end;
		}
	});
    int i = 1;
	while (i < intervals.size()){
	    Interval cur = intervals.get(i-1);
		if(intervals.get(i).start <= cur.end){
			cur.end = Math.max(intervals.get(i).end, cur.end);
			intervals.remove(i);
		}else{
			cur = intervals.get(i);
			i++;
		}
	}
	return intervals;
}
public class Solution {
    public int sumNumbers(TreeNode root) {
        if (root == null) return 0;
        return help(root, 0);
        
        
    }
    private int help(TreeNode root, int sum){
        if(root == null) return 0;
        if(root.left == null && root.right == null) return sum*10 + root.val;
        int val = 0;
        sum = sum*10 + root.val;
        val += help(root.left, sum);
        val += help(root.right, sum);
        return val;
    }
} public int integerBreak(int n) {
        int[] dp = new int[n+1];
        dp[1] = 1;
        for(int i = 2; i<=n; i++){
            for(int j=1; j*2<=i; j++){
                dp[i] = Math.max(dp[i], Math.max(j,dp[j])*Math.max(i-j, dp[i-j]));
            }
        }
        return dp[n];
    }

  public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        int n = A.length;
        if(n == 0) return 0;
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int i = 0; i<n; i++){
            for(int j = 0; j<n; j++){
                int sum1 = A[i] + B[j];
                map.put(-sum1, map.getOrDefault(-sum1, 0)+1);
            }
        }
        int res = 0;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                int sum2 = C[i] + D[j];
                res += map.getOrDefault(sum2,0);
            }
        }
        return res;
    }
public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if(nums.length < 4) return res;
        Arrays.sort(nums);
        for(in
        	t i = 0; i< nums.length -3; i++){
            if(i>0 && nums[i] == nums[i-1]) continue;
            for(int j = i+1; j<nums.length -2; j++){
                if(j>i+1 && nums[j] == nums[j-1]) continue;
                int left = j+1;
                int right = nums.length -1;
                while(left < right){
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if(sum > target) right--;
                    else if(sum < target) left++;
                    else{
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        left++;
                        right--;
                        while(left < right && nums[left] == nums[left -1]) left++;
                        while(left < right && nums[right] == nums[right + 1]) right--;
                    }
                }
            }
        }
      return res;  
    }

 public int threeSumClosest(int[] nums, int target) {
        int res = 0;
        if(nums.length <= 3) {
            for(int num : nums){
                res += num;
            }
            return res;
        }
        Arrays.sort(nums);
        res = nums[0] + nums[1] + nums[nums.length -1 ];
        for(int i = 0; i< nums.length-2; i++){
            int start = i+1;
            int end = nums.length -1;
            while(start < end){
            int sum = nums[i] + nums[start] + nums[end];
            if(sum == target) return sum;
            if(Math.abs(target -sum) < Math.abs(target -res)) res = sum;
            if(sum < target) start++;
            else end--;
            }
            
        }
       return res; 
    }

 public boolean isSubsequence(String s, String t) {
        if(s.length() == 0) return true;
        int m = s.length();
        int n = t.length();
        int i = 0;
        for(int j = 0; j < n; j++){
            if(s.charAt(i) == t.charAt(j)){
                i++;

            }
            if(i == m) return true;
        }
        return false;
    }

public class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        if(head == null) return null;
        if(head.next == null) return new TreeNode(head.val);
        ListNode fast = head;
        ListNode slow = head;
        ListNode pre = null;
        while(fast !=null && fast.next != null){
            fast = fast.next.next;
            pre = slow;
            slow = slow.next;
        }
        if(pre != null) pre.next = null;
        TreeNode root = new TreeNode(slow.val);
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(slow.next);
        return root;
        
    }
 
}
class UnionFind {
    private int[] father = null;
    private int count;
    private int find(int x){
        if(father[x] == x){
            return x;
        }
        return father[x] = find(father[x]);
    }
    public UnionFind(int n){
        father = new int[n];
        for(int i =0; i < n; i++){
            father[i] = i;
        }
    }
    public void connect(int a, int b){
        int roota = find(a);
        int rootb = find(b);
        if(roota != rootb) {
            father[roota] = rootb;
            count--;
        }
    }
    public int query() {
        return count;
    }
    public void set_count(int total) {
        count = total;
    }
}
public class Solution{
    public int numIslands(boolean[][] grid) {
        int count = 0;
        int n = grid.length;
        if(n==0) return 0;
        int m = grid[0].length;
        if(m==0) return 0;
        UnionFind union_find = new UnionFind(n*m);
        int total = 0;
        for(int i = 0; i < n; i++){
            for(int j = 0; j<m; j++){
                if(grid[i][j] == 1) total++;
            }
        }
        union_find.set_count(total);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(grid[i][j] == 1){
                    if(i>0 && grid[i-1][j] == 1){
                        union_find.connect(i*m + j, (i-1)*m + j);
                    }
                    if(i < n-1 && grid[i+1][j] == 1){
                        union_find.connect(i*m + j, (i+1)*m + j);
                    }
                    if(j > 0 && grid[i][j-1] == 1) {
                        union_find.connect(i*m + j, i*m + j-1);
                    }
                    if(j < m - 1 && grid[i][j+1] == 1){
                        union_find.connect(i*m + j, i*m + j+1);
                    }
                }

            }
        }
        return union_find.query();
    }
}
public class Solution {
public int numIslands(char[][] grid){
    if(grid == null || grid.length ==0 || grid[0].length ==0) return 0;
    int m = grid.length;
    int n = grid[0].length;
    int[] root = new int[m*n];
    int count = 0;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(grid[i][j] == '1'){
                root[i*n +j] = i*n + j;
                count++;
            }
        }
    }
    int[] dx = {-1, 1, 0, 0};
    int[] dy = {0, 0, -1, 1};
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(grid[i][j] == '1'){
                for(int k = 0; k<4; k++){
                    int x = i + dx[k];
                    int y = j + dy[k];
                    if(x >= 0 && x<m && y >= 0 && y < n && grid[x][y] == '1'){
                        int croot = getRoot(root, i*n + j);
                        int nroot = getRoot(root, x*n + y);
                        if(nroot != croot){
                            root[croot] = nroot;
                            count--;
                        }
                    }
                }
            }
        }
    }
    return count;
}
    public int getRoot(int[] arr, int i){
        while(arr[i] != i){
            i = arr[arr[i]];
        }
        return i;
    }

}
public int numDecodings(String s){
    if(s.length() == 0 || s == null) return 0;
    int n = s.length();
    int[] dp = new int[n+1];
    dp[0] = 1;
    dp[1] = s.charAt(0) != '0'  1 : 0;
    for(int i = 2; i < n+1; i++){
        if(s.charAt(i-1) != '0')
            dp[i] += dp[i-1];
        int val = Integer.valueOf(s.substring(i-2,i));
        if(val >= 10 && val <= 26)
            dp[i] += dp[i-2];
    }
    return dp[n];
}
public int numIslands (char[][] grid){
    if(grid.length == 0 || grid[0].length == 0 || grid == null)
        return 0;
    int m = grid.length;
    int n = grid[0].length;
    int count = 0;
    for(int i = 0; i < m; i++){
        for(int j = 0; j<n;j++){
            if(grid[i][j] == '1'){
                count++;
                merge(grid, i, j);
            }
        }
    }
    return count;
}

public void merge(char[][] grid, int i, int j){
    int m = grid.length;
    int n = grid[0].length;
    if(i<0 || j<0 || i>=m ||j>=n || grid[i][j] != '1') return;
    grid[i][j] = '0';
    merge(grid, i,j-1);
    merge(grid, i-1, j);
    merge(grid, i, j+1);
    merge(grid, i+1, j);
}



















