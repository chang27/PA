//frog jump: one iteration, one rescursion

public boolean canCross(int[] stones) {
        if(stones.length <= 1) return true;
        Map<Integer, Set<Integer>> map = new HashMap<>();
        map.put(0, new HashSet<>());
        map.get(0).add(1);
        for(int i = 1; i < stones.length; i++) {
            map.put(stones[i], new HashSet<>());
        }
        for(int i = 0; i < stones.length - 1; i++){
            for(int step : map.get(stones[i])){
                int reach = step + stones[i];
                if(reach == stones[stones.length - 1]){
                    return true;
                }
                if(map.containsKey(reach)){
                    if(step - 1 > 0) map.get(reach).add(step - 1);
                    map.get(reach).add(step);
                    map.get(reach).add(step + 1);
                }
            }
        }
        return false;
    }

   //To print out all the results:
  public List<List<Integer>> canCross(int[] stones) {
            List<List<Integer>> res = new ArrayList<>();
            List<Integer> cur = new ArrayList<>();
            cur.add(stones[0]);
            helper(stones, 0, 0, res, cur);
            return res;
        }
        private void helper(int[] stones, int index, int k, List<List<Integer>> res, List<Integer> cur){
            if(index == stones.length - 1){
               res.add(new ArrayList<>(cur));
               return;
            }
            for(int i = k - 1; i <= k + 1; i++){
                int next = find(stones, stones[index] + i, index + 1);
                if(next > 0) {
                    cur.add(stones[next]);
                    helper(stones, next, i, res, cur);
                    cur.remove(cur.size() - 1);
                }
            }
           // return false;
        }
        private int find(int[] nums, int val, int start){
            int s = start;
            int e = nums.length - 1;
            while(s <= e){
                int mid = s + (e - s)/2;
                if(nums[mid] > val){
                    e = mid - 1;
                }
                else if(nums[mid] < val) {
                    s = mid + 1;
                }else return mid;
            }
            return -1;
        }


// this is tricky: [1, 2, 3, 9] sums up to 6 for first 3 elements. cover to 6.
// for 7 need to add a patch 7 to reach 13. With 7, all numbers inside 13 can be reached.
// then add 9, can reach 22.

  public int minPatches(int[] nums, int n) {
        long sum = 0;
        int i = 0; 
        int patch = 0;
        while(sum < n){
            if( i >= nums.length || nums[i] > sum + 1){
                patch++;
                sum += sum + 1;
            }else{
                sum += nums[i];
                i++;
            }
        }
        return patch;
        
    }
    // delete dp[i][j] = dp[i-1][j] + cost of delete
    // insert dp[i][j] = dp[i][j-1] + cost of insert
     public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for(int i = 0; i <=m; i++){
            dp[i][0] = i;
        }
        for(int i = 0; i <= n; i++){
            dp[0][i] = i;
        }
        for(int i = 0; i < m ; i++){
           for(int j = 0; j < n; j++){
               if(word1.charAt(i) == word2.charAt(j)){
                   dp[i+1][j+1] = dp[i][j];
               }else{
                   dp[i+1][j+1] = Math.min(dp[i][j] + 1, Math.min(dp[i+1][j] + 1, dp[i][j+1] + 1));
               }
           }
        }
        return dp[m][n];
        
    }

    class Solution {
    public List<Point> outerTrees(Point[] points) {
        if(points.length < 2) return Arrays.asList(points);
        sortPolar(points, bottomLeft(points));
        Stack<Point> stack = new Stack<>();
        stack.push(points[0]);
        stack.push(points[1]);
        for(int i = 2; i < points.length; i++){
            Point cur = stack.pop();
            while(ccw(stack.peek(), cur, points[i]) < 0){
                cur = stack.pop();
            }
            stack.push(cur);
            stack.push(points[i]);
        }
        return new ArrayList<>(stack);
        
    }
    // a,b,c  ccw < 0 : clockwise; > 0 counterclockwise; vector (ab) & (bc)
    
    private int ccw(Point a, Point b, Point c){
        return (b.x - a.x)*(c.y - b.y) - (c.x - b.x)*(b.y - a.y);
        
    }
    private int dist(Point a, Point b){
        return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
    }
    private Point bottomLeft(Point[] points){
        Point bl = points[0];
        for(int i = 1; i < points.length; i++){
            if (points[i].y < bl.y || (points[i].y == bl.y && points[i].x < bl.x)){
                bl = points[i];
            }
        }
        return bl;
    }
    private void sortPolar(Point[] points, Point r) {
    	//for the largest polar, reverse the order with farest first.
        Arrays.sort(points, (a, b) -> {
            int cmp = ccw(a, r, b);
            int cmd = dist(a, r) - dist(b, r);
            return cmp == 0? cmd: cmp;
        });
        Point s = points[0];
        Point e = points[points.length -1];
        int i = points.length - 2;
        while(i >= 0 && ccw(s, points[i], e) == 0) i--;
        int l = i + 1;
        int h = points.length - 1;
        while(l < h){
            Point tmp = points[l];
            points[l] = points[h];
            points[h] = tmp;
            l++;
            h--;
        }
    }
    
}


class Solution {
    public List<Point> outerTrees(Point[] points) {
        if(points.length <= 3) return Arrays.asList(points);
	    Arrays.sort(points, (a, b) -> a.x == b.x ? a.y - b.y : a.x - b.x);
	    List<Point> ans = new ArrayList<>();
	    for(int i = points.length - 1; i >= 0; i--) {
			while(ans.size() > 1 && ccw(ans.get(ans.size() - 2), ans.get(ans.size() -1), points[i]) < 0){
				ans.remove(ans.size() - 1);

			}
			ans.add(points[i]);
	}
	if(ans.size() == points.length) return ans;

	for(int i = 1; i < points.length ; i++){
		while(ans.size() > 1 && ccw(ans.get(ans.size() - 2), ans.get(ans.size() -1), points[i]) < 0) {
			ans.remove(ans.size() - 1);
		}
		ans.add(points[i]);
	}
	ans.remove(ans.size() - 1);
	return ans;
       
        
    }
    private int ccw(Point a, Point b, Point c){
        return (b.x - a.x)*(c.y - b.y) - (c.x - b.x)*(b.y - a.y);
        
    }    
}

//Sort the critical line and use a treeMap to maintain the rectangular at this line.
public List<int[]> getSkyline(int[][] buildings) {
	List<int[]> heights = new ArrayList<>();
	for(int[] b : buildings) {
		heights.add(new int[]{b[0], -b[2]});
		heights.add(new int[]{b[1], b[2]});

	}
	Collections.sort(heights, (a,b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
	TreeMap<Integer, Integer> tm = new TreeMap<>(Collections.reverseOrder());
	int pre = 0;
	tm.put(0, 0); // that is for all the rectangulars finish at the critical line 
	List<int[]> res = new ArrayList<>();
	for(int[] h :heights){
		if(h[1] < 0){
			tm.put(-h[1], tm.getOrDefault(-h[1], 0) + 1);
		}else{
			if(tm.get(h[1]) == 1) tm.remove(h[1]);
			else tm.put(h[1], tm.get(h[1]) - 1);
		}
		if(tm.isEmpty()){
			res.add(new int[]{h[0], 0});
		}
		int cur = tm.firstKey();
		if(pre != cur){
			res.add(new int[]{h[0], cur});
			pre = cur;
		}

	}
	return res;

}


multiplication table: totally largest m*n value.
low <= target <= high: garantee to find the value inside the table.
 public int findKthNumber(int m, int n, int k) {
        int l = 1;
        int h = m*n;
        while(l < h){
            int mid = l + (h - l)/2;
            int c = count(mid, m, n);
            if(c >= k) h = mid;//key step: h may in the table or not. But can still use left and right to find it.
            else l = mid + 1;
        }
        return h;
    }
    private int count(int value, int m, int n){
        int count = 0;
        for(int i = 1; i<= m; i++){
            int tmp = Math.min(value/i, n);
            count += tmp;
        }
        return count;
    }

 public boolean searchMatrix(int[][] matrix, int target) {
        if(matrix == null ||matrix.length == 0 || matrix[0].length == 0) return false;
        int row = 0;
        int col = matrix[0].length - 1;
        while(row < matrix.length && col >= 0){
            if(matrix[row][col] == target) return true;
            if(matrix[row][col] > target) col--;
            else row++;
        }
        return false;
    }


      public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length;
        int n = dungeon[0].length;
        int[] dp = new int[n+1];
        Arrays.fill(dp, Integer.MAX_VALUE); // key step;
        dp[n-1] = 1;
        for(int i = m-1; i >= 0; i--) {
            for(int j = n-1; j >= 0; j--) {
                int val = Math.min(dp[j], dp[j + 1]) - dungeon[i][j];
                dp[j] = Math.max(1, val);
            }
        }
        return dp[0];
    }

 public double[] medianSlidingWindow(int[] nums, int k) {
       double[] res = new double[nums.length - k + 1];
       // Key point: for comparator to avoid overflow: use nums[a] < nums[b] instead of nums[a] - nums[b].
        TreeSet<Integer> small = new TreeSet<Integer>((a,b) -> nums[a] == nums[b] ? a - b: nums[a] < nums[b] ? -1 : 1);
        TreeSet<Integer> large = new TreeSet<Integer>((a,b) -> nums[a] == nums[b] ? a - b: nums[a] < nums[b] ? -1 : 1);
    	for(int i = 0; i < nums.length; i++) {
		    if(i >= k) {
			    if(!small.remove(i - k)){
				    large.remove(i - k);
			    }
		}
		if(small.size() <= large.size()) {
				large.add(i);
				int tmp = large.first();
				large.remove(tmp);
				small.add(tmp);
			
		}else{
			small.add(i);
			int t = small.last();
			small.remove(t);
			large.add(t);
		}

		if(i >= k -1) {
			res[i - k + 1] = k%2 == 0? ((double)nums[small.last()] + (double)nums[large.first()])/2.0 : (double)nums[small.last()];
		}
	}
	return res;
    }
class MedianFinder {
    private PriorityQueue<Integer> small;
    private PriorityQueue<Integer> large;
    /** initialize your data structure here. */
    public MedianFinder() {
       small  = new PriorityQueue<>((a, b) -> b - a);
       large  = new PriorityQueue<>();
        
        
    }
    
    public void addNum(int num) {
        if(small.size() <= large.size()) {
            large.offer(num);
            small.offer(large.poll());
        }else{
            small.offer(num);
            large.offer(small.poll());
        }
        
    }
    
    public double findMedian() {
        if(small.size() > large.size()) return (double)small.peek();
        else return ((double)small.peek() + (double)large.peek())/2.0;
        
    }
}

 public int numDistinct(String s, String t) {
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[n+1][m+1];
        for(int i = 0; i < m; i++){
            dp[0][i] = 1; // empty string can be a subsequence of any string only once.
        }
        for(int j = 0; j < n; j++){
            for(int i = 0; i < m; i++){
                if(s.charAt(i) == t.charAt(j)){
                    dp[j+1][i+1] = dp[j+1][i] + dp[j][i]; 
                }else dp[j+1][i+1] = dp[j+1][i];
            }
        }
        return dp[n][m];
    }

 public int numDistinct(String s, String t) {
        int m = s.length();
        int n = t.length();
        int[] dp = new int[n+1];
        dp[0] = 1;
        
        for(int j = 0; j < m; j++){
            for(int i = n-1; i>= 0; i--){
                if(s.charAt(j) == t.charAt(i)){
                    dp[i+1] += dp[i]; 
                }
            }
        }
        return dp[];
    }

 public int checkRecord(int n){
    final int MOD = 1000000007;
 	int[][][] f = new int[n+1][2][3];
 	f[0] = new int[][]{{1,1,1},{1,1,1}};
 	for(int i = 1; i <= n; i++){
 		for(int j = 0; j <2; j++){
 			for(int k = 0; k < 3; k++){
 				int val = f[i-1][j][2]; // ...P
 				if(j > 0) val = (val + f[i-1][j-1][2])%MOD; //...A
 				if(k > 0) val = (val + f[i-1][j][k-1])%MOD; //...L
 				f[i][j][k] = val;
 			}
 		}
 	}
 	return f[n][1][2];// sequence long: n, at most 1 A, at most 2 L;
 }
法二：
  public int checkRecord(int n) {
    final int MOD = 1000000007;
 	long[] P = new long[n+1];
    long[] PorL = new long[n+1];
     P[0] = 1;
    PorL[0] = 1;
        P[1] = 1;
        PorL[1] = 2;
    for(int i = 2; i <=n; i++){
        P[i] = PorL[i-1];
        PorL[i] = (P[i] + P[i-1] + P[i-2])%MOD;
    }
    long res = (int)PorL[n];
    for(int i = 0; i < n; i++){
        res = (res + (PorL[i]*PorL[n-i-1]) % MOD) % MOD;
    }
    return (int)res;
    }


public ListNode reverseBetween(ListNode head, int m, int n) {
	ListNode dmy  = new ListNode(0);
	dmy.next = head;
	ListNode pre = dmy;
	for(int i = 1; i <m; i++){
		pre = pre.next;
	}
	ListNode tail = pre.next;
	ListNode next = tail;
	for(int i = m; i < n; i++) {
		pre.next = tail.next;
		tail.next = tail.next.next;
		pre.next.next = next;
		next = pre.next;
	}
	return dmy.next;
}

public ListNode reverseKGroup(ListNode head, int k){
	int n = 0;
	ListNode i = head;
	while(i != null){
		n++;
		i = i.next;
	}
	ListNode dmy = new ListNode(0);
	dmy.next = head;
	ListNode pre = dmy;
	ListNode tail = head;
	while(n >= k){
		for(int i = 1; i < k; i++){
			ListNode next = tail.next.next;
			tail.next.next = pre.next;
			pre.next = tail.next;
			tail.next = next;
			tail.next = next;
		}
		pre = tail;
		tail = tail.next;

		n -= k;
	}
	return dmy.next;
}

public int countRangeSum(int[] nums, int lower, int upper) {
	if(nums == null || nums.length ==0 || lower > upper) return 0;
	long[] presum = new long[nums.length + 1];
	for(int i = 0; i < nums.length; i++){
		presum[i+1] = presum[i] + nums[i];
	}
	return merge();
}

private int merge(long[] sum, int lower, int upper, int start, int end) {
		if(start >= end){
			return 0;
		}
		int mid = start + (end - start)/2;
		int cnt = merge(sum, lower, upper, start, mid) + merge(sum, lower, upper, mid+1, end);
		for(int i = start, k = mid + 1, j = mid + 1; i <= mid, i++){
			while(k <= end && sum[k] - sum[i] < lower){
				k++;
			}
			while(j <= end && sum[j] - sum[i] <= upper){
				j++;
			}
			cnt += (j - k);
		}
		ms(sum, start, mid, end);
		return cnt;
}

private void ms(long[] sum, int s, int m, int e) {
	long[] copy = new long[e - s + 1];
	int i = start, j = mid + 1, p = 0;
	while(i <= mid && j <= end){
		if(sum[i] <= sum[j]){
			copy[p] = sum[i];
			p++;
			i++;
		}else{
			copy[p] = sum[j];
			j++;
		}
	}
	while(i <= mid){
		copy[p++] = sum[i++];
	}
	while(j <= end) {
		copy[p++] = sum[j++];
	}
	System.arraycopy(copy, 0, sum, s, e-s + 1);
}


class Solution {
    public int reversePairs(int[] nums) {
        
        if(nums == null ||nums.length ==0) return 0;
        int res = 0;
        Node root = null ;
        for(int num : nums){
            res += search(root, (long)num * 2 + 1);
            root = insert(root, num);
        }
        return res;   
    }
    
    private int search (Node root, long val){
        if(root == null){ 
            return 0;
        }else if(root.val == val){
            return root.cnt;
        }else if(root.val > val) {
            return root.cnt + search(root.left, val);
        }else {
            return search(root.right, val);
        }
    }
    
    private Node insert(Node root, int val){
        if(root == null){
            root = new Node(val);
        }else if(root.val == val){
            root.cnt ++;    
        }else if(root.val > val){
            root.left = insert(root.left, val);   
        }else{
            root.cnt++;
            root.right = insert(root.right, val);
        }
        return root;
    }
class Node{
    Node left;
    Node right;
    int cnt;
    int val;
    public Node(int val){
        this.val = val;
        this.cnt = 1;
    }
}
    
}

 public int reversePairs(int[] nums) {
        if(nums == null || nums.length == 0) return 0;
        return merge(nums, 0, nums.length - 1);
}
    private int merge(int[] nums, int start, int end){
        if(start >= end) return 0;
        int mid = start + (end - start)/2;
        int cnt = merge(nums, start, mid) + merge(nums, mid + 1, end);
        int[] copy = new int[end - start + 1];
        int k = mid + 1, p = 0;
        for(int i = start,j = mid + 1; i <= mid; i++) {
            while(j <= end && nums[i] > 2L * nums[j]) j++;
            cnt += j - mid - 1;
            while(k <= end && nums[k] < nums[i]) copy[p++] = nums[k++];
            copy[p++] = nums[i];
        }
        while(k <= end) copy[p++] = nums[k++];
        System.arraycopy(copy, 0, nums, start, copy.length);
        return cnt;
    }

     public List<String> addOperators(String num, int target) {
        	List<String> res = new ArrayList<>();
    	StringBuilder sb = new StringBuilder();
    	dfs(num, target, sb, res, 0, 0, 0);
        return res;
    }

    private void dfs(String num, int target, StringBuilder sb, List<String> res, int pos, long pre, long mult){
    	if(target == pre && pos == num.length()){
    		res.add(sb.toString());
    		return;
    	}
    	for(int i = pos; i < num.length() ; i++) {
    		if(num.charAt(pos) == '0' && i > pos) break; //for continuous 0s;
    		long cur = Long.parseLong(num.substring(pos, i+1));
    		int len = sb.length();
    		if(pos == 0){
    			dfs(num, target, sb.append(cur), res, i+1, cur, cur);
                sb.setLength(len); //for start from beginning but contains more than 1 digits as one number!
    		}else{
    			dfs(num, target, sb.append("+").append(cur), res, i+1, pre + cur, cur);
    			sb.setLength(len);
    			dfs(num, target, sb.append("-").append(cur), res, i+1, pre - cur, -cur);
    			sb.setLength(len);
    			dfs(num, target, sb.append("*").append(cur), res, i+1, pre - mult + mult*cur, mult*cur);
    			sb.setLength(len);
    		}
        }
        
    }


   public int maximumGap(int[] nums) {
        if(nums == null || nums.length < 2) return 0;
        int min = nums[0];
        int max = nums[0];
        for(int i = 1; i < nums.length; i++) {
            min = Math.min(min, nums[i]);
            max = Math.max(max, nums[i]);
        }
        if(min == max) return 0;
        int gap = (int)Math.ceil((double)(max - min)/(nums.length - 1)); //maxGap no less than gap.
        //so number of bucket = nums.length - 1 can guarantee that elements inside one bucket less than gap.
        int bucketNum = (int)((double)(max - min)/gap + 1);
        int[] bucketMin = new int[bucketNum];
        int[] bucketMax = new int[bucketNum];
        Arrays.fill(bucketMin, Integer.MAX_VALUE);
        Arrays.fill(bucketMax, Integer.MIN_VALUE);
        for(int num : nums) {
            if(num == max || num == min) continue;
            int index = (num - min)/gap;
            bucketMin[index] = Math.min(bucketMin[index], num);
            bucketMax[index] = Math.max(bucketMax[index], num);
        }
        int res = 0;
        int pre = min;
        for(int i = 0; i < bucketNum; i++){
            if(bucketMin[i] == Integer.MAX_VALUE){
                continue;
            }
            res = Math.max(res, bucketMin[i] - pre);
            pre = bucketMax[i];
        }
        res = Math.max(res, max - pre);
        return res;
    }


//decode ways ii
 class Solution {
    public int numDecodings(String s) {
        int MOD = 1000000007;
        if(s == null || s.length() == 0) return 0;
        long pre = ways(s.charAt(0));
        if(s.length() < 2) return (int) pre;
        long cur = ways(s.charAt(0), s.charAt(1)) + pre*ways(s.charAt(1));
        for(int i = 2; i < s.length(); i++) {
            long tmp = cur;
            cur = (cur*ways(s.charAt(i)) + pre*ways(s.charAt(i-1), s.charAt(i)))%MOD;
            pre = tmp;
        }
        return (int)cur;
        
    }
    private int ways(char c) {
        if(c == '*'){
            return 9;
        }else if(c == '0'){
            return 0;
        }
        return 1;
    }
    
    private int ways(char c1, char c2) {
        if(c1 == '*' && c2 =='*') {
            return 15;
        }else if(c1 != '*' && c2 != '*') {
            String str = "" + c1 + "" + c2;
            if(Integer.parseInt(str) >= 10 && Integer.parseInt(str) <= 26){
                return 1;
            }
        }else if(c1 == '*'){
            String str1 = "" + c2;
            if(Integer.parseInt(str1) >= 0 && Integer.parseInt(str1) <= 6){
                return 2;
            }else return 1;
        }else if(c2 == '*'){
            if(c1 == '1') return 9;
            else if(c1 =='2') return 6;
        }
        return 0;
    }
    
    
}

   public List<Integer> largestDivisibleSubset(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if(nums == null || nums.length == 0) return res;
        Arrays.sort(nums);
        int[][] dp = new int[nums.length][2];
        for(int i = 0; i < nums.length; i++){
            dp[i] = new int[]{1, -1};
        }
        int max = 0;
        int maxIndex = -1;
        for(int m = 0; m < nums.length; m++) {  // here need to begin from index 0 for the situation when there is just one element in the array. 
            for(int n = m -1; n >= 0; n--) { 
                if(nums[m] % nums[n] == 0){
                    if(dp[m][0] < dp[n][0] + 1){
                        dp[m][0] = dp[n][0] + 1;
                        dp[m][1] = n;
                    }
                }
            }
            if (max < dp[m][0]){
                max = dp[m][0];
                maxIndex = m;
            }
        }
        int i = maxIndex;
        while(i >= 0) {
            res.add(0, nums[i]);
            i = dp[i][1];
        }
        return res;
        
    }

List<Integer> ans;
public List<Integer> largestDivisibleSubset(int[] nums) {
	List<Integer> res = new ArrayList<>();
	if(nums == null || nums.length == 0) return res;
	Arrays.sort(nums);
	int max = 1;
	dfs(res, nums, max, start);
}

private void dfs(List<Integer> res, int[] nums, int max, int s) {

	if(res.size() > max) {
		max = result.size();
		ans = new ArrayList<Integer>(res);
	}

	if(s == nums.length){
		return;
	}
	for(int i = start; i < nums.length; i++){
		if(res.size() == 0) {
			res.add(nums[i]);
			dfs(res, nums, max, i+1);
			res.remove(res.size() -1);
		}else{
			if(nums[i]% res.get(res.size() -1) == 0) {
				res.add(nums[i]);
				dfs(res, nums, max, i+ 1);
				res.remove(res.size() - 1);
			}
		}
	}
}


   public int numberOfArithmeticSlices(int[] A) {
        if(A == null || A.length < 3) return 0;
        Map<Integer, Integer>[] dp = new Map[A.length];
        int res = 0;
        for(int i = 0; i <A.length; i++){
            dp[i] = new HashMap<Integer, Integer>();
            for(int j = 0; j < i; j++) {
                long diff = (long) A[i] - A[j]; // 不能在（A[i] - A[j]）加括号，越界。
                if(diff > Integer.MAX_VALUE || diff < Integer.MIN_VALUE){
                    continue;
                } 
                if(dp[j].containsKey((int)diff)){
                    res += dp[j].get((int)diff); // new counted one is a sequence of length 2.
                    dp[i].put((int)diff, dp[i].getOrDefault((int)diff, 0) + dp[j].get((int)diff) + 1);
                }else {
                    dp[i].put((int)diff, dp[i].getOrDefault((int)diff, 0) + 1);
            }
        }
        }
        return res;
    }

    public int findIntegers(int num) {
        if(num == 0) return 1;
        StringBuffer sb = new StringBuffer(Integer.toBinaryString(num));
        int n = sb.length();
        int[] a = new int[n];
        int[] b = new int[n];
        //a represents the numbers with ending 0
        //b represents the numbers with ending 1
        a[0] = 1;
        b[0] = 1;
        for(int i = 1; i < n; i++) {
            a[i]= a[i-1] + b[i-1];
            b[i] = a[i - 1];
        }
        int res = a[n-1] + b[n-1]; // res represents all possible results for binary number with length n;
        for(int i = 1; i < n-1 ; i++){
            if(sb.charAt(i) == '1' && sb.charAt(i-1) == '1') break; // all results are less than this number with two "11" as the most significant bits;
            if(sb.charAt(i) == '0' && sb.charAt(i+1) == '0') res -= b[n-i-2];
        }
        return res;
    }

 private boolean doSolve(char[][] board, int row, int col) {
        for (int i = row; i < 9; i++, col = 0) { // note: must reset col here!
            for (int j = col; j < 9; j++) {
                if (board[i][j] != '.') continue;
                for (char num = '1'; num <= '9'; num++) {
                    if (isValid(board, i, j, num)) {
                        board[i][j] = num;
                        if (doSolve(board, i, j + 1))
                            return true;
                        board[i][j] = '.';
                    }
                }
                return false;
            }
        }
        return true;
    }
    
  private boolean isValid(char[][] board, int row, int col, char c){
      int newr = (row / 3) * 3;
      int newc = (col / 3) * 3;
      for(int i = 0; i < 9; i++){
          if(board[i][col] == c || board[row][i] == c || board[newr + i / 3][newc + i % 3] == c){ // !!!! no 'c'!!!!!
              return false;
          }
      }
      return true;
  }
}

 public int strangePrinter(String s) {
        if(s == null || s.length() == 0) return 0;
        int n = s.length();
        int[][] dp = new int[s.length()][s.length()];
        for(int i = 0; i < n; i++) {
            dp[i][i] = 1; // same character just need print once.
        }
        for(int i = n-2; i >= 0; i--){
            for(int j =  i+ 1; j < n; j++){
                dp[i][j] = dp[i+1][j] + 1;// add one more character
                for(int k = i+1; k <= j; k++){
                    if(s.charAt(i) == s.charAt(k)){
                        dp[i][j] = Math.min(dp[i][j], dp[i][k-1] + dp[k][j] - 1);
                    }
                }
            }
            
        }
        return dp[0][n-1];
        
    }

public String removeDuplicateLetters(String s) {
        if(s.length() < 2) return s;
        int[] cnt = new int[26];
        StringBuilder sb = new StringBuilder();
        while(s.length() > 0){
        
        for(int i = 0; i < s.length(); i++){
            cnt[s.charAt(i) - 'a'] += 1;    // every time after shorten the string need to recalculate the number of chars.
        }
        int pos = 0;
        for(int i = 0 ; i < s.length(); i++){ // here need to start from the beginning, just in case the first element is the single character which need to be outputed directly.
            if(s.charAt(i) < s.charAt(pos)){
                pos = i;
            }
            if(--cnt[s.charAt(i) - 'a'] == 0) break; // record the smallest position before the single char.
        }
         sb.append(s.charAt(pos));
        s = s.substring(pos + 1).replaceAll("" + s.charAt(pos), "");
        Arrays.fill(cnt, 0);// need to recalculate the number 
        }
       return sb.toString(); 
    }

   public String removeDuplicateLetters(String s) {
        if(s.length() < 2) return s;
        Map<Character, Integer> map = new HashMap<>();
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), i); // record the last position of every distinct character;
        }
         int start = 0;
        for(int i = 0; i < map.size(); i++) {
            int end = findMinPos(map);
            char min = 'z' + 1;
            for(int j = start; j <= end; j++) {
                if(map.containsKey(s.charAt(j)) && s.charAt(j) < min){
                    min = s.charAt(j);
                    start = j + 1;
                }
            }
            sb.append(min);
            map.remove(min);
        }
        return sb.toString();
    }



public String numberToWords(int num) {
	if(num == 0) return "Zero";
	String[] less20 = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
	String[] tywords = {"Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
	String[] dexwords = {"Billion", "Million", "Thousand", "Hundred"};
	int[] radix = {1000000000, 1000000, 1000, 100};
	StringBuilder sb = new StringBuilder();
	for(int i = 0; i < radix.length; i++) {
		int cnt = num / radix[i];
		if(cnt == 0) continue;
		sb.append(numberToWords(cnt) + " ");
		sb.append(dexwords[i] + " ");
		num = num % cnt;
	}
	if(num < 20){
		sb.append(less20[num]);
	}else{
		sb.append(tywords[num/10 - 2] + " ");
		sb.append(less20[num % 10]);
	}
	return sb.toString.trim();
}



  public int firstMissingPositive(int[] nums) {
        for(int i = 0; i < nums.length ; i++){
           while(nums[i] <= nums.length && nums[i] >0 && nums[i] != nums[nums[i]-1]){// firstly see if place i  and place nums[i] - 1 share the same value, if so will end in infinite loops. Second, use while to make sure continues elements can be placed in the right place.
               swap(nums, i, nums[i] -1);
           }
        }
        for(int i = 0; i < nums.length; i++){
            if(nums[i] != i+1){
                return i+1;
            }
        }
     return nums.length + 1;
    }
    private void swap(int[] nums, int i, int j){
        if( i != j) {
            nums[i] ^= nums[j];
            nums[j] ^= nums[i];
            nums[i] ^= nums[j];
        }
    }

    class Solution {
    public boolean isScramble(String s1, String s2) {
        if(s1.length() != s2.length()) return false;
        if(s1.equals(s2)) return true;
        int n = s1.length();
        int[] cnt = new int[26];
        Arrays.fill(cnt, 0);
        for(int i = 0; i < n; i++){
            cnt[s1.charAt(i) - 'a']++;
            cnt[s2.charAt(i) - 'a']--;
        }
        for(int i = 0; i < 26; i++){
            if(cnt[i] != 0) return false;
        }
        for(int i = 1; i <= n-1; i++){
            if(isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) return true;
            if(isScramble(s1.substring(0, i), s2.substring(n - i)) && isScramble(s1.substring(i), s2.substring(0, n - i))) return true;
        }
        return false;
    }
}

//Key: always remove the largest index of the substituting element
// keep all the position information for duplicate elements.
class RandomizedCollection {
        List<Integer> nums;
        Map<Integer, LinkedHashSet<Integer>> map;
        Random ran = new Random();
    /** Initialize your data structure here. */
    public RandomizedCollection() {
        nums = new ArrayList<>();
        map = new HashMap<>();
    }
    
    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        boolean exist = map.containsKey(val);
        if(!exist){
            map.put(val, new LinkedHashSet<Integer>()); // linked hashset O(1) to remove.
        }
        int size = nums.size();
        map.get(val).add(size);
        nums.add(val);
        return !exist ;
        
    }
    
    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val)) return false;
        int idx = map.get(val).iterator().next();
        
        map.get(val).remove(idx);
        
        if(idx < nums.size() - 1) { // if idx is nums.size() - 1, no need to do these things.
            int tmp = nums.get(nums.size() - 1);
            nums.set(idx, tmp);
            map.get(tmp).remove(nums.size() - 1);
            map.get(tmp).add(idx);
        }
        nums.remove(nums.size() - 1);
        if(map.get(val).isEmpty()) map.remove(val);
        return true;
    }
    
    /** Get a random element from the collection. */
    public int getRandom() {
        return nums.get(ran.nextInt(nums.size()));
        
    }
}

class Solution {
    int max = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        if(root == null) return 0;
        helper(root);
        return max;
        
    }
    private int helper(TreeNode root) {
        if(root == null) return 0;
        int left = Math.max(0, helper(root.left));
        int right = Math.max(0, helper(root.right));
        max = Math.max(max, left + right + root.val);
        return Math.max(left, right) + root.val;
    }
}
 

     Stack<String> stack = new Stack<>();
        for(int i = 0; i < code.length();){
            if(i>0 && stack.isEmpty()) return false;
            if(code.startsWith("<![CDATA[", i)){
                int j = i+9;
                i = code.indexOf("]]>", j);
                if(i < 0) return false;
                i += 3;
            }else if(code.startsWith("</", i)){
                int j = i + 2;
                i = code.indexOf('>', j);
                if(i < 0 || i == j || i - j > 9) return false;
                for(int k = j; k < i; k++){
                    if(!Character.isUpperCase(code.charAt(k))) return false;
                }
                String s = code.substring(j, i++);
                if(stack.isEmpty() || !stack.pop().equals(s)) return false;
            }else if(code.startsWith("<", i)){
                int j = i + 1;
                i = code.indexOf('>', j);
                if(i < 0 || i == j || i - j > 9) return false;
                for(int k = j; k < i; k++){
                    if(!Character.isUpperCase(code.charAt(k))) return false;
                }
                String s = code.substring(j, i++);
                stack.push(s);
            }else{
                i++;
            }
        }
        return stack.isEmpty();

 if(heights == null || heights.length == 0) return 0;
        int maxarea = 0;
        int area = 0;
        Deque<Integer> stack = new LinkedList<>();
        for(int i = 0; i < heights.length;) {
            if(stack.isEmpty() || heights[stack.peekFirst()] <= heights[i]){
                stack.offerFirst(i++);
            }else{
                int top = stack.pollFirst();
                if(stack.isEmpty()){
                    area = heights[top]*i;
                }else{
                    
                 area = heights[top]*(i - stack.peekFirst() -1);
                }
                if(area < maxarea){
                    maxarea = area;
                }
            }
        }
        while(!stack.isEmpty()) {
            int top = stack.pollFirst();
            if(stack.isEmpty()){
                area = heights[top]*heights.length;
            }else{
                area = heights[top]*(heights.length - stack.peekFirst() - 1);
            }
            if(area > maxarea){
                maxarea = area;
            }
        }
        return maxarea;
    }
}


public int getMaxRepetitions(String s1, int n1, String s2, int n2) {
        int len1 = s1.length(), len2 = s2.length();
	int[] rcnt = new int[len2 + 1];
	int[] nIdx = new int[len2 + 1];
    Arrays.fill(rcnt, 0);
    Arrays.fill(nIdx, 0);
	int m = 0, cnt = 0;
	for(int i = 1; i <= n1; i++) {
		for(int j = 0; j < s1.length(); j++){
			if(s2.charAt(m) == s1.charAt(j)){
				m++;
				if(m == len2){
					m = 0;
					cnt++;
				}
			}

		}
		rcnt[i] = cnt;
		nIdx[i] = m;
		for(int k = 0; k < i; k++){
			if(nIdx[k] == m){
				int pre = rcnt[k];
				int cur = (n1 - k)/(i - k)*(rcnt[i] - rcnt[k]); // here need to firstly calculate (n1-k)/(i - k)
				int suf = (rcnt[(n1 - k)%(i - k) + k] - rcnt[k]);
				return (pre + cur + suf) / n2;
			}
		}

	}
	return rcnt[n1]/ n2;
    }

 class Node {
    int key;
    int val;
    Node pre;
    Node next;
    public Node (int key, int val) {
        this.key = key;
        this.val = val;
    }
}
public class LRUCache {
    Map<Integer, Node> map;
    int capacity;
    Node head = null;
    Node tail = null;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>();
    }
    
    public int get(int key) {
        if(map.containsKey(key)) {
            Node n = map.get(key);
            remove(n);
            setHead(n);
            return n.val;
        }
        return -1;
    }
    private void remove(Node n) {
        if(n.pre != null){
            n.pre.next = n.next;
        }else{
            head = n.next;
        }
        
        if(n.next != null){
            n.next.pre = n.pre;
        }else{
            tail = n.pre;
        }
    }
    
    private void setHead(Node n) {
            n.next = head;
            n.pre = null;
        if(head != null) {
            head.pre = n;
            
        }
        head = n;
        if(tail == null){
            tail = head;
        }
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)) {
            Node n = map.get(key);
            n.val = value;
            remove(n);
            setHead(n);
        }else{
            Node newNode = new Node(key, value);
            if(map.size() >= capacity) {
                map.remove(tail.key);// Here is very important, must firstly remove tail.key then remove tail.
                remove(tail);
                
                setHead(newNode);
            }else{
                setHead(newNode);
            }
            map.put(key, newNode);
        }
    }
}

public class LFUCache {
    class Node{
        int freq = 1;
        Node pre;
        Node next;
        Set<Integer> set;
        public Node(){
            set = new LinkedHashSet<>();
        }
    }
    Map<Integer, Integer> vmap;
    Map<Integer, Node> nmap;
    int capacity;
    Node head;
    public LFUCache(int capacity) {
        this.capacity = capacity;
        vmap = new HashMap<>();
        nmap = new HashMap<>();
        head = null;
    }
    
    public int get(int key) {
       if(vmap.containsKey(key)) {
        increase(key);
        return vmap.get(key);
        }
        return -1;
    }
    
     public void put(int key, int value) {
        if(capacity == 0) return;
        if(vmap.containsKey(key)){
            vmap.put(key, value);
            increase(key);
        }else{
            if(vmap.size() >= capacity){
                removeLeaseFreqKey();
            }
            vmap.put(key,value);
            addToHead(key);
        }
    }
    private void addToHead(int key) {
       if(head == null) {
           head = new Node();
          head.set.add(key);
       }else if(head.freq > 1){
           Node newNode = new Node();
           newNode.set.add(key);
           newNode.next = head;
           head.pre = newNode;
           newNode.pre =null;
           head = newNode;
       }else{
           head.set.add(key);
       }
       nmap.put(key, head);
    }
    private void removeLeaseFreqKey(){
        if (head == null) return ;
        int old = 0;
        for(int n : head.set) {
            old = n;
            break;
        }
        head.set.remove(old);
        if(head.set.size() == 0){
            remove(head);
        }
        vmap.remove(old);
        nmap.remove(old);
    }
    private void remove(Node node){
        if(node.pre != null){
            node.pre = node.next;
        }else{
            head = node.next;
        }
        if(node.next != null){
            node.next.pre = node.pre;

        }
    }
    private void increase (int key) {
        Node node = nmap.get(key);
        int freq = node.freq;
        node.set.remove(key);
        if(node.next != null && node.next.freq == freq + 1) {
            
            node.next.set.add(key);
            
        }else{
            Node newNode = new Node();
            newNode.freq = node.freq + 1;
            newNode.set.add(key);
            if(node.next != null){
                node.next.pre = newNode;
            }
            newNode.next = node.next;
            node.next = newNode;
            newNode.pre = node;
//             if(node.next == null){
//                 node.next = newNode;
//                 newNode.pre = node;
//             }else{
//                 newNode.next = node.next;
//                 node.next.pre = newNode;
//                 node.next = newNode;
//                 newNode.pre = node;
                
//             }
           
        }
        nmap.put(key, node.next);
        if(node.set.size() == 0) remove(node);// remember to remove the empty node;
    }
    // private void insertNext(int key, Node node){
    //     Node newNode = new Node();
    //     newNode.freq = node.freq + 1;
    //     newNode.set.add(key);
    //     node.set.remove(key);
    //     if(node.next != null){
    //         node.next.pre = newNode;
    //     }
    //     newNode.next = node.next;
    //     node.next = newNode;
    //     newNode.pre = node;
    //     nmap.put(key,newNode);
    // }
    
    
}

public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> res = new ArrayList<>();
        int i = 0;
        while(i < intervals.size() && intervals.get(i).end < newInterval.start){
            res.add(intervals.get(i));
            i++;
        }
        while(i < intervals.size() && intervals.get(i).start <= newInterval.end) {
            newInterval = new Interval(Math.min(intervals.get(i).start, newInterval.start), Math.max(intervals.get(i).end, newInterval.end));
            i++;
        }
        res.add(newInterval);
        while(i < intervals.size()) res.add(intervals.get(i++));
        return res;
    }

class Solution {
    public List<List<Integer>> palindromePairs(String[] words) {
        List<List<Integer>> res = new ArrayList<>();
        if(words == null || words.length == 0) return res;
        Map<String, Integer> map = new HashMap<>();
        for(int i = 0; i < words.length; i++) {
            map.put(words[i], i);
        }
        for(int i = 0; i < words.length; i++) {
            for(int j = 0; j <= words[i].length(); j++) {  // empty string is also a valid palindrome
                String str1 = words[i].substring(0, j);
                String str2 = words[i].substring(j);
                if(isPalind(str1)) {
                    String str2rev = new StringBuilder(str2).reverse().toString();
                    if(map.containsKey(str2rev) && map.get(str2rev) != i){    // need to see not itsself!
                            res.add(new ArrayList<>(Arrays.asList(map.get(str2rev), i)));
                    }
                }
                if(isPalind(str2) && str2.length() != 0){ // avoid duplicates!
                    String str1rev = new StringBuilder(str1).reverse().toString();
                    if(map.containsKey(str1rev) && map.get(str1rev) != i) {
                        res.add(new ArrayList<>(Arrays.asList(i, map.get(str1rev))));
                    }
                }
            }
        }
        return res;
        
    }
    private boolean isPalind (String ss){
        int s = 0;
        int e = ss.length() - 1;
        while(s < e){
            if(ss.charAt(s++) != ss.charAt(e--)) return false;
        }
        return true;   
    }
}

// KMP table means: if we reach here, where to backup if next character can not match
private int[] getTable(String s) {
	int[] table = new int[s.length];
	int i = 0;
	for(int j = 1; j < s.length();) {
		if(s.charAt(i) == s.charAt(j)) {
			table[j]  = ++i; // length and next to try element
			j++;
		}else{
			if(i > 0) {
				i = table[i -1]; // i can not 
				, table[i-1] indicate next element to try
			}else {
				table[j] = i;
				j++;
			}
		}
	}
}

public int findNumberOfLIS(int[] nums) {
	int n = nums.length, res = 0, maxlen = 1;
	int[] cnt = new int[n];

}

 class TrieNode{
        TrieNode[] child;
        String word;
        public TrieNode(){
            this.child = new TrieNode[26];
        }
    }
    private TrieNode buildTrie(String[] word){
        TrieNode root = new TrieNode();
        for(String s : word){
            TrieNode p = root;
        for(int i = 0; i < s.length(); i++){
            int j = s.charAt(i) - 'a';
            if(p.child[j] == null) p.child[j] = new TrieNode();
            p = p.child[j];
        }
        p.word = s;
    }
        return root;
    }
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        TrieNode root = buildTrie(words);
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                dfs(root, i, j, board, res);
            }
        }
        return res;
    }
    private void dfs(TrieNode root, int i, int j, char[][] bd, List<String> res) {
            char c = bd[i][j];
            if(c == '*' || root.child[c - 'a'] == null){
                return;
            }
            root = root.child[c - 'a'];
            if(root.word != null){
                res.add(root.word);
                root.word = null; //remove duplicates
            }
            bd[i][j] = '*';
            if(j+1 <bd[0].length) dfs(root, i, j+1, bd, res);
            if(i > 0) dfs(root, i - 1, j, bd, res);
            if(j > 0) dfs(root, i, j-1, bd, res);
            if(i + 1 < bd.length) dfs(root, i+1, j, bd, res);
            bd[i][j] = c;
    }

     public int candy(int[] ratings) {
        int sum = 0;
        int[] candy = new int[ratings.length];
        int n = ratings.length;
        Arrays.fill(candy, 1);
        for(int i = 1; i < n; i++) {
            if(ratings[i] > ratings[i-1]){
                candy[i] = candy[i-1] + 1;
            }
        }
        for(int i = n-2; i >= 0; i--){
            if(ratings[i] > ratings[i+1] && candy[i] <= candy[i+1]){
                candy[i] = candy[i+1] + 1;
            }
        }
        for(int c : candy){
            sum += c;
        }
        return sum;
    }

    class Solution {
    int[][] dir = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    public int cutOffTree(List<List<Integer>> forest) {
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>(){
                public int compare(int[]a, int[]b){
                    return a[2] - b[2];
                }
            });
            for(int i = 0; i < forest.size(); i++){
                for(int j = 0; j < forest.get(0).size(); j++) {
                    int f = forest.get(i).get(j);
                    if(f > 1){
                        
                    pq.offer(new int[]{i, j, forest.get(i).get(j)});
                    }
                }
            }
        int sum = 0;
        int[] start = new int[2];
        while(!pq.isEmpty()){
            int[] target = pq.poll();
            int step = bfs(forest, start, target);
            if(step == -1) return -1;

            sum += step;
            start = target;

        }
            return sum;
        
    }
     private int bfs(List<List<Integer>> forest, int[] start, int[]end){
            boolean[][] visited = new boolean[forest.size()][forest.get(0).size()];
            Queue<int[]> q = new LinkedList<>();
            q.offer(start);
            visited[start[0]][start[1]] = true; // must have a visited table for BFS!!!
            int step = 0;
            while(!q.isEmpty()){
                for(int i = q.size() -1; i>= 0; i--) {
                    int[] s = q.poll();
                    if(s[0] == end[0] && s[1] == end[1]) return step;
                    for(int[] d : dir) {
                        int j = s[0] + d[0];
                        int k = s[1] + d[1];
                        if(j < 0 || k < 0 || j >= forest.size() || k >= forest.get(0).size()|| forest.get(j).get(k) == 0 || visited[j][k])continue;
                        q.offer(new int[]{j, k});
                        visited[j][k] = true; // must change the visited status here!!!
                    }
                }
                step++;
            }
            return -1;
        }
    
}

public boolean isSelfCrossing(int[] x) {
	int l = x.length;
	if(l <= 3) return false;
	for(int i = 3; i < l; i++){
		if(x[i] >= x[i-2] && x[i-1] <= x[i-3]) return true; //cross with three lines before!
		if(x >= 4 && x[i-1] == x[i-3] && x[i] + x[i-4] >= x[i-2]) return true; // cross with four lines before!
		if(x >= 5 && x[i] + x[i-4] >= x[i-2] && x[i-2] >= x[i-4] && x[i-1] <= x[i-3] && x[i-1] + x[i-5] >= x[i-3]) return true; // cross with five lines before!


	}
	return false;
}

class Solution {
    public int findKthNumber(int n, int k) {
        int cur = 1;
        k = k -1;
        while(k > 0){
            int step = calStep(n, cur, cur + 1);
            if(step <= k) {
                k -= step;
                cur++;
            }else{
                cur *= 10;
                k--;
            }
        }
        return cur;     
    }
    
    private int calStep(int n, long n1, long n2){
        int sum = 0;
        while(n1 <= n){
            sum += Math.min(n+1, n2) - n1;   // that is to count the steps level by level. n2 is the right bound of cur+1 level downwards
            n1 *= 10;
            n2 *= 10;
        }
        return sum;
    }
}


class Solution {
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int n = nums1.length;
        int m = nums2.length;
        int[] res = new int[k];
        for(int i = Math.max(0, k-m); i <= k && i <= n; i++){
            int[] part1 = get(nums1, i); // i indicates the range;
            int[] part2 = get(nums2, k-i);
            int[] cadi = merge(part1, part2, k);
            if(greater(cadi, res, 0, 0)){
                res = cadi;
            }
        }
        return res;
        
    }
    
    private int[] merge(int[] n1, int[] n2, int size){
        int[] res = new int[size];
        int m = 0;
        int n = 0;
        for(int i = 0; i < size; i++){
            if(greater(n1, n2, m,n)){
                res[i] = n1[m++];
                
            }else res[i] = n2[n++];
        }
        return res;
    }
    
    private boolean greater(int[] n1, int[] n2, int i, int j){
        while(i < n1.length && j < n2.length && n1[i] == n2[j]) {
            i++;
            j++;
        }
        return j == n2.length || i < n1.length && n1[i] > n2[j];
    }
    
    private int[] get(int[] nums, int size){
        int[] res = new int[size];
        int last = -1;
        for(int i = 0; i < size; i++) {  // (size - i) including j. j + size-i -1 < length; 
            for(int j = last + 1; j + size - i -1 < nums.length; j++) {
                if(nums[j] > res[i]){
                    res[i] = nums[j];
                    last = j;
                }
            }
        }
        return res;
    }
}



public boolean isNumber(String s) {
        s = s.trim();
        boolean eseen = false;
        boolean pointseen = false;
        boolean numberseen = false;
        for(int i = 0; i < s.length(); i++) {
            if(s.charAt(i) >= '0' && s.charAt(i) <= '9'){
                if(numberseen == false){
                    numberseen = true;
                }
            }else if(s.charAt(i) == '.'){
                if(eseen || pointseen){
                    return false;
                }
                pointseen = true;
            }else if(s.charAt(i) == '+' || s.charAt(i) == '-'){
                if(i != 0 && s.charAt(i-1) != 'e'){
                    return false;
                }
            }else if(s.charAt(i) == 'e'){
                if(!numberseen || eseen){
                    return false;
                }
                eseen = true;
                numberseen = false;
            }else return false;
        }
        return numberseen;
    }

    public boolean isMatch(String s, String p) {
        int n1 = s.length();
        int n2 = p.length();
        boolean[][] dp = new boolean[n1+ 1][n2 + 1];
        dp[0][0] = true;
        for(int i = 2; i <= n2; i++){
            dp[0][i] = (p.charAt(i-1) == '*' && dp[0][i-2]);
        }
        for(int i = 1; i <= n1; i++){
            for(int j = 1; j <= n2; j++) {
                if(p.charAt(j-1) != '*'){
                    dp[i][j] = dp[i-1][j-1] && (s.charAt(i-1) == p.charAt(j-1) || p.charAt(j-1) == '.');
                }else if(p.charAt(j-1) == '*'){
                    dp[i][j] = dp[i][j-2] || (dp[i-1][j] && (s.charAt(i-1) == p.charAt(j-2) || p.charAt(j-2) == '.'));
                }
            }
        }
        return dp[n1][n2];
    }

    class Solution {
    public int maxPoints(Point[] points) {
        int l = points.length;
        if(l<= 2) return l;
        int res = 2;
        
        for(int i = 0; i < l-1; i++){
            int ovrlap = 0;
            int max = 0;
            Map<String, Integer> map = new HashMap<>();
            for(int j = i+1; j < l; j++){
                int x = (points[j].x - points[i].x);
                int y = (points[j].y - points[i].y);
                if(x == 0 && y == 0) {
                    ovrlap++;
                    continue;
                }
                int gcd = gcd(x, y);
                x /= gcd;
                y /= gcd;
                String key = x + ":" + y;
                if(map.containsKey(key)){
                    map.put(key, map.get(key) + 1);
                }else{
                    map.put(key, 1);
                }
                max = Math.max(max, map.get(key));
            }
            
            res = Math.max(res, max + ovrlap + 1);
        }
        return res;
        
    }
    private int gcd(int x, int y){
        if(y == 0) return x;
        return gcd(y, x%y);
    }
}

class Solution {
       public String nearestPalindromic(String n) {
           Long num = Long.parseLong(n);
           Long h = findHigher(num + 1);
           Long l = findLower(num - 1);
           if(Math.abs(num - h) < Math.abs(num - l)){
            return String.valueOf(h);
           }
           return String.valueOf(l);
       }
    
    private Long findHigher(Long num) {
        String n = String.valueOf(num);
        char[] arr = n.toCharArray();
        for(int i = 0; i < n.length()/2 ; i++){
            arr[n.length() -i - 1] = arr[i];
        }
        if(Long.parseLong(String.valueOf(arr)) < num){
            for(int i = (n.length() - 1)/2; i >= 0; i--){
                if(++arr[i] > '9'){
                    arr[i] = '0';
                }else{
                    break;
                }
            }
            for(int i = 0; i < n.length()/2 ; i++){
                arr[n.length() -i - 1] = arr[i];
            }     
        }
        return Long.parseLong(String.valueOf(arr)); 
    }
    
    private Long findLower(Long num) {
        String n = String.valueOf(num);
        char[] arr = n.toCharArray();
        for(int i = 0; i < n.length()/2 ; i++){
            arr[n.length() -i - 1] = arr[i];
        }    
        if(Long.parseLong(String.valueOf(arr)) > num) {

            for(int i = (n.length() -1)/ 2; i >= 0; i--) {
                if(--arr[i] <'0'){
                    arr[i] = '9';
                }else break;
            }
            if(arr[0] == '0'){
                char[] tmp = new char[arr.length - 1];
                Arrays.fill(tmp, '9');
                arr = tmp;
            }else{
                for(int i = 0; i < n.length()/2 ; i++){
                arr[n.length() -i - 1] = arr[i];
            }   
            }
        }
        return Long.parseLong(String.valueOf(arr)); 

    }
}

class Solution {
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        if(words == null || words.length == 0) {
            res.add("");
            return res;
        } 
        int i = 0;
        while(i < words.length){
            int len = -1; // last word do not need any spaces
            int w = i;
            for(; w < words.length && len + words[w].length() +1 <= maxWidth; w++){
                len += words[w].length() + 1;
            }
            int num = w - i;
            int even = 1;
            int more = 0;
            StringBuilder sb = new StringBuilder(words[i]);
            if(num > 1 && w != words.length){
                even += (maxWidth - len)/(num - 1);
                more = (maxWidth - len)%(num - 1);
            }
            for(int j = i+1; j < w; j++){
                
                for(int k = 0; k < even; k++){
                    sb.append(' ');
                }
                if(more > 0){
                    sb.append(' ');
                    more--;
                }
                sb.append(words[j]);
            }
            while(maxWidth > sb.length()){
                sb.append(' ');
            }
            res.add(sb.toString());
            i = w;
        }
        return res;
        
    }
}



class Solution {
    public int strongPasswordChecker(String s) {
       int a = 1, A = 1, d = 1;
        int res = 0;
        char[] ch = s.toCharArray();
        int[] arr = new int[ch.length];
        for(int i = 0; i < ch.length;) {
            if(Character.isDigit(ch[i])) d = 0;
            else if(Character.isLowerCase(ch[i])) a = 0;
            else if(Character.isUpperCase(ch[i])) A = 0;
            int j = i;
            while(i < ch.length && ch[i] == ch[j]) {
                i++;
            }
            arr[j] = i - j;
        }
        int missing = a + A + d;
        if(s.length() < 6){
                res += missing + Math.max(0, 6- (missing + s.length()));
        }else{
            int over = Math.max(0, ch.length - 20);
            int left = 0;
            res += over;
            if(over > 0){
            for(int k = 1; k < 3; k++){
                for(int i = 0; i < arr.length && over > 0; i++){
                    if(arr[i] < 3 || arr[i]%3 != (k-1)) continue;
                    arr[i] = arr[i] - Math.min(k, over);
                    over -= k;
                }
            }
            
                for(int i = 0; i < arr.length; i++){
                    if(arr[i] >= 3 && over> 0) {
                        int need = arr[i] - 2;
                        arr[i] -= Math.min(need, over);
                        over -= need;
                        
                    }
                }
            }
            
            for(int i = 0; i < arr.length; i++){
                left += arr[i]/3;
            }
            res += Math.max(missing, left);
            
        }
        return res;
        
        
        
    }
}


public boolean judgePoint24(int[] nums) {
        List<Double> list = new ArrayList<>();
        for(int num : nums) {
            list.add((double) num);

        }

        return  helper(list);


    }

    private boolean helper(List<Double> list){
            if(list.isEmpty()) return false;
            if(list.size() == 1 && Math.abs(list.get(0) - 24.0) <= 0.001){
                return true;
            }
            for(int i = 0; i < list.size() - 1; i++) {
                for(int j = i+1; j < list.size(); j++) {
                    Double a = list.get(i);
                    Double b = list.get(j);
                    List<Double> tmp = new ArrayList<>();
                    tmp.addAll(Arrays.asList(a-b, b-a, a*b, a+b));
                    if(Math.abs(b) > 0.001) tmp.add(a/b);
                    if(Math.abs(a) > 0.001) tmp.add(b/a);
                     list.remove(j);
                    list.remove(i);
                   
                    for(Double d : tmp){
                        list.add(d);
                        if(helper(list)) return true;
                        list.remove(d);
                    }
                    list.add(i, a);
                    list.add(j, b);
                   // list.add(j, b);
                  //  list.add(i, a);
                    
                    
                 }
            }
            return false;
    }

class Solution {
    public boolean wordPatternMatch(String pattern, String str) {
        if(pattern.length() == 0 && str.length() == 0) return true;
        if(pattern.length() == 0 || str.length() == 0) return false;
        Map<Character, String> map = new HashMap<>();
        Set<String> set = new HashSet<>();
        return helper(pattern, str, 0, 0, map, set);
        
    }
    
    private boolean helper(String pattern, String str, int pin, int sin, Map<Character, String> map, Set<String> set){
        if(pin == pattern.length() && sin == str.length()) return true;
        if(pin >= pattern.length() || sin >= str.length()) return false;
        
        for(int k = sin + 1; k <= str.length(); k++){
            String sub = str.substring(sin, k);
            if(!map.containsKey(pattern.charAt(pin)) && ! set.contains(sub)) {
                map.put(pattern.charAt(pin), sub);
                set.add(sub);
                if(helper(pattern, str, pin + 1, k, map, set)){
                    return true;
                }
                map.remove(pattern.charAt(pin), sub);
                set.remove(sub);
            }else if(map.containsKey(pattern.charAt(pin)) && map.get(pattern.charAt(pin)).equals(sub)){
                if(helper(pattern, str, pin + 1, k, map, set)){
                    return true;
                }
            }
        }
        return false;
    }
}

private void dfs(int[][] maze, int[] start, int[]destination, int[][] map, String path, int direct, int cnt) {
        if(cnt > map[start[0]][start[1]] || cnt > minStep) return;
        if(direct != -1){
            switch(direct){
                case 0: path += "u";
                        break;
                case 1: path += "d";
                        break;
                case 2: path += "l";
                        break;
                case 3: path += "r";
                        break;
            }
            int x = start[0];
            int y = start[1];
            map[x][y] = Math.min(map[x][y], cnt);
            while(isValid(maze, x+ dir[direct][0], y + dir[direct][1])){
                x += dir[direct][0];
                y += dir[direct][1];
                cnt++;
                map[x][y] = Math.min(map[x][y], cnt);
                if(x == destination[0] && y == destination[1]){
                    if(cnt < minStep){
                        minStep = cnt;
                        minS = path;
                    }else if(cnt == minStep && path.compareTo(minS) < 0){
                        minS = path;
                    }
                    return;
                }
            }
            start = new int[]{x, y};   
        }
        
        for(int i = 0; i < 4; i++){
            if(i == direct) continue;
            int[] cur = new int[]{start[0] + dir[i][0], start[1] + dir[i][1]};
            if(isValid(maze, cur[0], cur[1])){
                dfs(maze, start, destination, map, path, i, cnt);
            }
        }
        
        
    }










