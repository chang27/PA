public class Solution {
    public List<Integer> diffWaysToCompute(String input) {
        if(input == null || input.length() ==0) return new ArrayList<Integer>();
        Map<String, List<Integer>> map = new HashMap<String,List<Integer>>();
        return dfs(input, map);
        
        
    }
    private List<Integer> dfs(String in, Map<String,List<Integer>> map){
        
        if(map.containsKey(in)) return map.get(in);
        List<Integer> res = new ArrayList<Integer>();
        for(int i = 0; i< in.length(); i++){
            char c = in.charAt(i);
            if(c == '+' || c == '-' || c == '*'){
                String left = in.substring(0, i);
                String right = in.substring(i+1);
                List<Integer> ll = dfs(left, map);
                List<Integer> rr = dfs(right, map);
                for(Integer l : ll){
                    for(Integer r : rr){
                        Integer cr = 0;
                        switch(c){
                            case '+':
                                cr = l + r;
                                break;
                            case '-':
                                cr = l - r;
                                break;
                            case '*':
                                cr = l * r;
                                break;
                        }
                        res.add(cr);
                    }
                }
                
            }
        }
        if(res.size() == 0){
            res.add(Integer.parseInt(in));
        }
        map.put(in, res);
        return res;
    }
}
//quick select. quick sort
public class Solution {
    public int findKthLargest(int[] nums, int k) {
        k = nums.length - k;
        int lo = 0;
        int hi = nums.length - 1;
        while(true){
        int j = partition(nums, lo, hi);
        if(j > k) {
            hi = j-1;
        }else if(j < k){
            lo = j + 1;
            
        }else break;
        
    }
    return nums[k];
}
    public int partition(int[] nums, int lo, int hi){
        int w = lo+1;
        int v = hi;
        while(true){
        while(w < hi + 1 && lessthan(nums[w], nums[lo])){
            w++;
        }
        while((v > lo) && (nums[lo] <= nums[v])){
            v--;
        }
        if(w >= v) break;
        swap(nums, w, v);
        }
        swap(nums, lo, v);
        return v;
    }
    
    public void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
    public boolean lessthan (int x, int y){
        return x < y;
    }
}
 public int findMinArrowShots(int[][] points) {
        if (points == null || points.length ==0) return 0;
        Arrays.sort(points, (a,b)-> a[0] == b[0]? a[1] - b[1] : a[0] - b[0]);
        int arrow = 1;
        int limit = points[0][1];
        for(int i = 0; i< points.length; i++){
            if(points[i][0] > limit){
                arrow++;
                limit = points[i][1];
            }else{
                limit = Math.min(limit, points[i][1]);
            }
        }
        return arrow;
    }
 public int findMinArrowShots(int[][] points) {
        if (points == null || points.length ==0) return 0;
        Arrays.sort(points, new Comparator<int[]>(){
            public int compare(int[] a, int[] b){
                return a[1] - b[1];
            }
        });
        int arrow = 1;
        int limit = points[0][1];
        for(int i = 1; i< points.length; i++){
            if(points[i][0] > limit){
                arrow++;
                limit = points[i][1];
            }
        }
        return arrow;
    }
    //sort by the end point;
      public int eraseOverlapIntervals(Interval[] intervals) {
        if (intervals.length == 0 || intervals == null){
            return 0;
        }
        Arrays.sort(intervals, new Comparator<Interval>(){
            public int compare(Interval a, Interval b){
                return a.end - b.end;
            }
            
        });
        int cnt = 1;
        int limit = intervals[0].end;
        for(int i = 1; i < intervals.length; i++){
            if(intervals[i].start >= limit){
                limit = intervals[i].end;
                cnt++;
            }
        }
        
        return intervals.length - cnt;
    }
 public ListNode oddEvenList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode odd = head;
        ListNode even = head.next;
        ListNode ehead = even;
        while(even != null && even.next != null){
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = ehead;
        return head;
    }
public class Solution {
    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode slow = head;
        ListNode fast = head;
        while(fast.next != null && fast.next.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode right = sortList(slow.next);
        slow.next = null;
        ListNode left = sortList(head);
        ListNode cur = merge(left,right);
        return cur;
    }
    public ListNode merge(ListNode head1, ListNode head2){
        ListNode dummy = new ListNode(0);
        ListNode tmp = dummy;
        while(head1 != null && head2 != null){
            if(head1.val < head2.val){
                tmp.next = head1;
                head1 = head1.next;
            }else
            {
                tmp.next = head2;
                head2 = head2.next;
            }
            tmp = tmp.next;
            }
            if(head1 != null){
                tmp.next = head1;
            }else if(head2 != null){
                tmp.next = head2;
            }
            return dummy.next;
        }
}
// tips: use dummy head to simplify the list head;
// since the inserted node depends on the former node in dummy list, so use p.next to compare with the node to be inserted.
 public ListNode insertionSortList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode dummy = new ListNode(0);
        while(head != null){
            ListNode node = dummy;
            while(node.next != null && node.next.val < head.val){
                node = node.next;
            }
            ListNode tmp = head.next;
            head.next = node.next;
            node.next = head; 
            head = tmp;
        }
        return dummy.next;
        
    } 
 public ListNode deleteDuplicates(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode dummy = new ListNode(head.val == 0? 1:0);
        dummy.next = head;
        ListNode pre = dummy;
        ListNode cur = head;
        ListNode first = dummy;
        while(cur != null && cur.next != null){
            if(pre.val != cur.val && cur.val != cur.next.val){
                first.next = cur;
                first = first.next;
            }
            pre = cur;
            cur = cur.next;
        }
        if(pre.val != cur.val){
            first.next = cur;
            first = first.next;
        }
      first.next = null;
        return dummy.next;
    }

public ListNode deleteDuplicates(ListNode head){
    if(head == null) return head;
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode pre = dummy;
    ListNode cur = head;
    while(cur != null){
        if(cur.next != null && cur.val == cur.next.val){
            cur = cur.next;
        }
        if(pre.next = cur){
            pre = pre.next;

        }else{
            pre.next = cur.next;
        }
        cur = cur.next;
    }
    return dummy.next;
}
public ListNode reverseBetween(ListNode head, int m, int n){
    if( m>=n || head == null){
        return head;
    }
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode node = dummy;
    for(int i = 1; i < m; i++){
        if(node == null) return null;
        node = node.next;
    }
    ListNode pre = node;
    ListNOde m = node.next;
    ListNode n = m, post = m.next;
    for(int i = m; i < n; i++){
        if(post == null){
            return null;
        }
        ListNode tmp = post.next;
        post.next = n;
        n= post;
        post = tmp;
    }
    m.next = post;
    pre.next = n;
    return dummy.next;
}
 public ListNode reverseBetween(ListNode head, int m, int n){
    ListNode newhead = new ListNode(0);
    newhead.next = head;
    ListNode tail1 = newhead;
    for(int i = 1; i< m; i++){
        tail1 = tail1.next;
    }
    ListNode tail = tail1.next;
    ListNode nextnode = tail;
    for(int i = m; i <n ; i++){
        tail1.next = tail.next;
        tail.next = tail.next.next;
        tail1.next.next = nextnode;
        nextnode = tail1.next;
    }
    return newhead.next;
}
//this recursion method has TLE problem.

public class Solution {
    public void reorderList(ListNode head) {
        if(head == null || head.next == null || head.next.next == null){
            head = head;
        }
        else{
        head = reverse(head);
        reorderList(head.next.next);
        }
        
        }
    
    public ListNode reverse(ListNode head){
        if(head == null || head.next == null) return head;
        ListNode fast = head;
        ListNode node = null;
        while(fast.next != null){
            node = fast;
            fast = fast.next;
        }
        ListNode m = head.next;
        head.next = fast;
        fast.next = m;
        node.next = null;
        return head;
        
    }
}

 public void reorderList(ListNode head) {
        if(head == null) return;

        ListNode slow = head, fast = head;
        while(fast.next != null && fast.next.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
//here just need to maintain two pointer, one is head before
    //reverse point, another is the starting node in reverse.
        ListNode part2 = slow;
        ListNode node = slow.next;
        while(node != null && node.next != null){
            ListNode next = node.next;
            node.next = next.next;
            next.next = part2.next;
            part2.next = next;

        }

        ListNode node1 = head;
        ListNode node2 = part2.next;
        while(node1 != part2){
          
            ListNode next1 = node1.next;
            ListNode next2 = node2.next;
            part2.next = next2;//here must have part.next = next2;
            //otherwise last node can not be connected.
            node1.next = node2;
            node2.next = next1;
            node1 = next1;
            node2 = next2;
        }
    
        }

// deep copy: https://www.kancloud.cn/kancloud/data-structure-and-algorithm-notes/73016

public RandomListNode copyRandomList(RandomListNode head){
    if(head == null) return null;
    RandomListNode cur = head;
    while(cur != null){
        RandomListNode newnode = new RandomListNode(cur.label);
        newnode.next = cur.next;
        cur.next = newnode;
        cur = cur.next.next;
    }

    cur = head;
    while(cur != null){
        if(cur.random != null){
            cur.next.random = cur.random.next;
        }
        cur = cur.next.next;
    }
    RandomListNode newHead = head.next;
    cur = head;
    while(cur != null){
        RandomListNode newNode = cur.next;
        cur.next = cur.next.next;
        cur = cur.next;
        if(newNode.next != null){
            newNode.next = newNode.next.next;
        }
    }
    return newHead;
}

public RandomListNode copyRandomList(RandomListNode head){
   RandomListNode dummy = new RandomListNode(0);
   RandomListNode cur = dummy;
   Map<RandomListNode, RandomListNode> map = new HashMap<>();
   while(head != null){
       RandomListNode tmp = null;
       if(map.containsKey(head)){
           tmp = map.get(head);
       }else{
           tmp = new RandomListNode(head.label);
           map.put(head, tmp);
           
       }
       cur.next = tmp;
       if(head.random != null){
           if(map.containsKey(head.random)){
               tmp.random = map.get(head.random);
           }else{
               tmp.random = new RandomListNode(head.random.label);
               map.put(head.random, tmp.random);
           }
           cur.next.random = tmp.random;
       }
       head = head.next;
       cur = cur.next;
   }
   return dummy.next;
 } 

 public UndirectedGraphNode cloneGraph(UndirectedGraphNode node){
    if(node == null) return node;
    ArrayList<UndirectedGraphNode> nodes = getNodes(node);
    HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<>();
    for(UndirectedGraphNode n : nodes){
        map.put(n, new UndirectedGraphNode(n.label));
    }
    for(UndirectedGraphNode n : nodes){
        UndirectedGraphNode newNode = map.get(n);
        for(UndirectedGraphNode neighbor : n.neighbors){
            UndirectedGraphNode newNeigbor = map.get(neighbor);
            newnode.neighbors.add(newNeigbor);
        }
    }
    return map.get(node);

}
public ArrayList<UndirectedGraphNode> getNodes(UndirectedGraphNode node){
    Queue<UndirectedGraphNode> queue = new LinkedList<UndirectedGraphNode>();
    HashSet<UndirectedGraphNode> set = new HashSet<>();
    queue.offer(node);
    set.add(node);
    while(!queue.isEmpty){
        UndirectedGraphNode cur = queue.poll();
        for(UndirectedGraphNode neighbor : cur.neighbors){
            if(!set.contains(neighbor)){
                set.add(neighbor);
                queue.offer(neighbor);
            }
        }
    }
    return new ArrayList<UndirectedGraphNode>(set);
}

public UndirectedGraphNode cloneGraph(UndirectedGraphNode node){
    if(node == null) return node;
    LinkedList<UndirectedGraphNode> q = new LinkedList<>();
    HashMap<UndirectedGraphNode,UndirectedGraphNode> map = new HashMap<>();
    UndirectedGraphNode n = new UndirectedGraphNode(node.label);
    q.add(node);
    map.put(node, n);
    while(!q.isEmpty){
        UndirectedGraphNode cur = q.pop();
        ArrayList<UndirectedGraphNode> neighbor = cur.neighbors;
        for(UndirectedGraphNode nei : neighbor){
            if(!map.containsKey(nei)){
                UndirectedGraphNode copy = new UndirectedGraphNode(nei.label);
                map.get(cur).neighbors.add(copy);
                map.put(nei, copy);
                q.add(nei);
            }
            else{
                map.get(cur).neighbors.add(map.get(nei));
            }

        }
    }
    return n;
}

//state machine:
//define 00: die to die
// 01: live to live
// 10: live to die
// 11: die to live
// mark the transvered cell as one of the four states, and change the state at then end at the same time.
public class Solution {
    public void gameOfLife(int[][] board) {
        int m = board.length;
        int n = board[0].length;
        int[] dx = {-1, -1, -1, 0, 1, 1, 1, 0};
        int[] dy = {-1, 0, 1, -1, -1, 1, 0, 1};
        for(int i = 0; i< m; i++){
            for(int j = 0; j < n; j++){
                int cnt = 0;
                for(int k = 0; k < 8; k++){
                    int x = i + dx[k];
                    int y = j + dy[k];
                    if(x >= 0 && x < m && y >= 0 && y < n && (board[x][y] == 1 || board[x][y] == 2)){
                        cnt++;
                    }
                }
                if(board[i][j] == 1 && (cnt < 2 || cnt > 3)){
                    board[i][j] = 2;
                
                }
                else if( board[i][j] ==0 && cnt == 3){
                    board[i][j] = 3;
                }
            }
           
            
        }
         for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    board[i][j] %= 2;
                }
            }
        
    }
}

public void setZeroes(int[][] natrix){
    boolean fr = false;
    boolean bc = false;
    for(int i = 0; i < matrix.length; i++){
        if(matrix[i][0] == 0) {
            fc = true;
            break;
        }
    }
    for(int j = 0. j < matrix[0].length; j++){
        if(matrix[0][j] == 0) {
            fr = true;
            break;
        }
    }
    for(int i = 1; i < matrix.length; i++){
        for(int j = 1; j < matrix[0].length; j++){
            if(matrix[i][j] == 0) {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }
    for(in i = 1; i < matrix.length; i++){
        for(int j = 1; j < matrix[0].length; j++){
            if(matrix[i][0] == 0 || matrix[0][j] == 0){
                matrix[i][j] = 0;
            }
        }
    }
    if(fr){
        for(int j = 0; j < matrix[0].length; j++){
            matrix[0][j] = 0;
        }
    }
    if(fc) {
        for(int i = 0; i < matrix.length; i++){
            matrix[i][0] = 0;
        }
    }

}

//dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1);
public static int findMaxForm(String[] strs, int m, int n){
    int[][] dp = new int[m+1][n+1];
    for(String str: strs){
        int[] count = count(str);
        for(int i = m; i>= count[0]; i--){
            for(int j = n; j>= count[1]; j--){
                dp[i][j] = Math.max(dp[i][j], dp[i - count[0]][j-count[1]]);
            }
        }

    }
    return dp[m][n];

}
private static int[] count(String s){
    int[] result = new int[2];
    for(int i = 0; i < s.length(); i++){
        result[s.charAt(i) - '0']++;
    }
    return result;
}


public class Solution {
    public int rob(int[] nums) {
        if(nums.length == 0 || nums == null) return 0;
        if(nums.length == 1) return nums[0];
        int a = rob2(Arrays.copyOfRange(nums, 0, nums.length-1));
        int b = rob2(Arrays.copyOfRange(nums, 1, nums.length));
        return Math.max(a,b);
        
    }
    private int rob2(int[] nums){
        int take = 0;
        int notake = 0;
        int max = 0;
        for(int num : nums){
            take = notake + num;
            notake = max;
            max = Math.max(take, notake);
        }
        return max;
    }
}
public int rob(TreeNode root) {
        int[] num = rob2(root);
        return Math.max(num[0], num[1]);
    }
private int[] rob2(TreeNode root){
        if(root == null) return new int[2];
        int[] left = rob2(root.left);
        int[] right = rob2(root.right);
        int[] res = new int[2];
        res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        res[1] = root.val + left[0] + right[0];
        return res;
    }

public class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        List<Integer> list = new ArrayList<Integer>();
        help(result, list, root, sum);
        return result;
    }
    private void help(List<List<Integer>> res, List<Integer> cur, TreeNode root, int sum){
        if(root == null) return;
        cur.add(root.val);
        sum = sum - root.val;
        if(root.left == null && root.right == null && sum == 0){
            res.add(new ArrayList(cur));
        }
        else{
            help(res, cur, root.left, sum);
            help(res, cur, root.right, sum);
        }
        cur.remove(cur.size()-1);
    }
}
public int minPathSum(int[][] grid) {
        if(grid.length == 0 || grid[0].length ==0 || grid == null) return 0;
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0]= grid[0][0];
        for(int i = 1; i < m; i++){
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        for(int j = 1; j < n; j++){
            dp[0][j] = dp[0][j-1] + grid[0][j];
        }
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
        }
    }
        return dp[m-1][n-1];
}

public int uniquePaths(int m, int n){
    int[][] map = new int[m][n];
    for(int i = 0 ; i < m; i++){
        map[i][0] = 1;
    }
    for(int j = 0; j < n; j++){
        map[0][j] = 1;
    }
    for(int i = 1; i < m; i++){
        for(int j = 1; j < n; j++){
            map[i][j] = map[i-1][j]+map[i][j-1];
        }
    }
    return map[m-1][n-1];
}
public int uniquePaths(int m, int n){
    if(m > n) return uniquePaths(n, m);
    int[] pre = new int[m];
    int[] cur = new int[m];
    Arrays.fill(pre, 1);
    Arrays.fill(cur, 1);
    for(int j = 1; j < n; j++){
        for(int i = 1; i < m; i++){
            cur[i] = cur[i-1] + pre[i];
        }
        int[] tmp = pre;
        pre = cur;
        cur = tmp;
    }
    return pre[m-1];
    }
public int uniquePaths(int m, int n){
    if (m > n) return uniquePaths(n, m);
    int[] cur = new int[m];
    Arrays.fill(cur, 1);
    for(int j = 1; j < n; j++){
        for(int i = 1; i < m; i++){
            cur[i] += cur[i-1];
        }
    }
    return cur[m-1];
}

public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int width = obstacleGrid[0].length;
    int[] dp = new int[width];
    dp[0] = 1;
    for(int i = 0; i < obstacleGrid.length; i++){
        for(int j = 0; j < width; j++){
            if(obstacleGrid[i][j] == 1){
                dp[j] = 0;
            }
            else if( j > 0){
                dp[j] += dp[j-1];
            }
        }
    }
    return dp[width - 1];
}
//以一个字母结尾的string length 是多少，就是说以这个字母结尾的substring 有多少个。
//那么我们知道以某个字符结束的最大字符串包含其他以该字符结束的字符串的所有子字符串，
public int findSubstringInWraproundString(String p) {
        int[] cnt = new int[26];
        int max = 0;
        for(int i = 0; i< p.length(); i++){
            if(i>0 &&((p.charAt(i) - p.charAt(i-1) == 1) || (p.charAt(i-1) - p.charAt(i) == 25))){
                max++;
            }
            else{
                max = 1;
            }
            int idx = p.charAt(i) - 'a';
            cnt[idx] = Math.max(cnt[idx], max);
        }
        int sum = 0;
        for(int i = 0; i < 26; i++){
            sum += cnt[i];   
        }
        return sum;
    }

public class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n+1];
        dp[0] = 0;
        for(int i = 1; i < n+1; i++){
            dp[i] = i;
        }
        
        for(int i = 1; i <= n; i++){
            for(int j = 1; j*j <=i; j++){
                dp[i] = Math.min(dp[i], dp[i-j*j] + 1);
            }
        }
        return dp[n];
    }
}

//generate larger ugly number from smaller ones:  
//Be careful about the cases such as 6, in which we need to forward both pointers of 2 and 3.


public int nthUglyNumber(int n){
    int[] res = new int[n];
    res[0] = 1;
    int k2 = 0, k3 = 0, k5 = 0;
    for(int i = 1; i<n; i++){
        res[i] = Math.min(2*res[k2], Math.min(3*res[k3], 5*res[k5]));
        if(res[i] == 2*res[k2]) k2++;
        if(res[i] == 3*res[k3]) k3++;
        if(res[i] == 5*res[k5]) k5++;

    }
    return res[n-1];
}

public int nthSuperUglyNumber(int n, int[] primes){
    int[] res = new int[n];
    res[0] = 1;
    int[] cur = new int[priimes.length];
    for(int i = 1; i < n; i++){
        res[i] = Integer.MAX_VALUE;
        for(int j = 0; j < primes.length; j++){
            if(primes[j] * res[cur[j]] == res[i-1]) {
                cur[j]++;
            }// 增加选中的prime， 并去除其他引起duplicate的项，因为已经选了最小值
            //在上一次循环里，所以不需要<=的条件 
            res[i] = Math.min(res[i], primess[j]*res[cur[j]] );
        }
    }
    return res[n-1];
}


public class Solution {
    public boolean makesqure(int[] nums) {
        if (nums == null || nums.length < 4) return false;
        int sum = 0;
        for(int num : nums){
            sum += num;
        }
        if(sum % 4 != 0) return false;
        Arrays.sort(nums);
        reverse(nums);
        int[] total = new int[4];
        return dfs(nums, total, 0, sum/4);
    }
    private boolean dfs(int[] nums, int[] sum, int index, int target){
        if(index >= nums.length) {
            if(sum[0] == sum[1] == sum[2] == target){
                return true;
            }
            return false;
        }
        for (int i =0; i < 4; i++){
            if(sum[i] + nums[index] > target) continue;
            sum[i] += nums[index];
            if (dfs(nums, sum, index + 1, target)) {
                return true;
            }
            sum[i] -= nums[index];
        }
        return false;
    }
    private void reverse(int[] nums) {
        int i = 0; 
        int j = nums.length -1;
        while( i < j) {
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }
}

public class Solution {
    public int maximalSquare(char[][] matrix) {
        if(matrix.length == 0 || matrix[0].length ==0 || matrix == null) return 0;
        int m = matrix.length;
        int n = matrix[0].length;
        int max = 0;
        int[] pre = new int[n+1];
        int[] cur = new int[n+1];
        for(int i= 1; i <= m; i++){
            for(int j = 1; j < n+1; j++){
                if(matrix[i-1][j-1] - '0' == 1){
                    cur[j] = Math.min(Math.min(pre[j-1], pre[j]), cur[j-1]) + 1;
                    max = Math.max(max, cur[j]);
                }
                else cur[j] = 0;
                
            }
            pre = Arrays.copyOf(cur, cur.length);
            Arrays.fill(cur, 0);
        }
       return max*max; 
    }
}

public class Codec{
    Map<String, String> shorturl = new HashMap<>();
    Map<String, String> longurl = new HashMap<>();
    static String base = "http://tinyurl.com/";
    public String encode(String longUrl){
        if(shorturl.containsKey(longUrl)){
            return base+shorturl.get(longUrl);
        }
        String charSet = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIGKLMNOPQRSTUVWXYZ";
        while(!shorturl.containsKey(longUrl)){

        }
    }
}



public class Codec {
    Map<String, String> short1 = new HashMap<>();
    Map<String, String> long1 = new HashMap<>();
    static String base = "http://tinyurl.com/" ;

    // Encodes a URL to a shortened URL.
    public String encode(String longUrl) {
        if(short1.containsKey(longUrl)){
            return base + short1.get(longUrl);
        }
        String lib = "abcdefghigklmnopqrstuvwxyz1234567890";
        while(!short1.containsKey(longUrl)){
            StringBuilder sb = new StringBuilder();
            for(int i = 0; i< 6; i++){
                int index = (int) (Math.random()*lib.length());
                sb.append(lib.charAt(index));
            }
            String value = sb.toString();
            if (!long1.containsKey(value)) {
                long1.put(value, longUrl);
                short1.put(longUrl, value);
            }
        }
        return base + short1.get(longUrl);
    }

    // Decodes a shortened URL to its original URL.
    public String decode(String shortUrl) {
        String key = shortUrl.replace(base, "");
        return long1.get(key);
        
    }
}
//#530
//This is an easy problem with general method:
public class Solution {
        TreeSet<Integer> set = new TreeSet<>();
        int min = Integer.MAX_VALUE;
    public int getMinimumDifference(TreeNode root) {
        //if(root == null) return min;
        //Stack<TreeNode> stack = new Stack<>();
            if(root == null) return min;
            if(!set.isEmpty()){
                if(set.ceiling(root.val) != null){
                    min = Math.min(min, set.ceiling(root.val)-root.val);
                }
                if(set.floor(root.val) != null){
                    min = Math.min(min, root.val - set.floor(root.val));
                }
                
            }
            set.add(root.val);
            getMinimumDifference(root.left);
            getMinimumDifference(root.right);
            
            
    
    return min;
}
}

public int getMinimumDifference(TreeNode root){
    int min = Integer.MAX_VALUE;
    Stack<TreeNode> stack = new Stack<>();
    TreeNode cur = root, pre = null;
    while(cur != null || !stack.isEmpty()){
        if(cur != null){
            stack.push(cur);
            cur = cur.left;
        }else{
            cur = stack.pop();
            if(pre != null){
                min = Math.min(cur.val - pre.val, min);
            }
            pre = cur;
            cur = cur.right;
        }


    }
    return min;
}
public Solution{
    int min = Integer.MAX_VALUE;
    TreeNode pre = null;
public int getMinimumDifference(TreeNode root){
       if (root == null) return min;
       getMinimumDifference(root.left);
       if(pre != null){
        min = Math.min(min, root.val - pre.val);
       }
       pre = root;
       getMinimumDifference(root.right);
       return min;
}
}

 public int findPairs(int[] nums, int k) {
        if(nums == null || nums.length == 0 || k < 0) return 0;
        Arrays.sort(nums);
        int res = 0;
        int i = 0;
        int j = 1;
        while(i < nums.length && j < nums.length) {
            if(j<=i || nums[j] < k + nums[i] ){
                j++;
            }
            else if(nums[i] + k < nums[j] || (i > 0 && nums[i] == nums[i-1])) {
                i++;
            }else{
                res++;
                i++;
            }
        }
        return res;
        
    }

public int findPairs(int[] nums, int k) {
        if(nums == null || nums.length == 0 || k < 0) return 0;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
        
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        for(Map.Entry<Integer, Integer> entry : map.entrySet()){
            if(k == 0){
                if (entry.getValue() >= 2) res++;
                
            }else{
                if(map.containsKey(entry.getKey() + k)) res++;
            }
        }
        return res;
    }
public class Solution {
    public int evalRPN(String[] tokens) {
        Stack<Integer> s = new Stack<>();
        for(String t : tokens){
            if (t.charAt(t.length() - 1) <= '9' && t.charAt(t.length()-1) >= '0') {
                s.push(Integer.valueOf(t));
            }else{
                int num1 = s.pop();
                int num2 = s.pop();
                switch(t){
                    case "+": s.push(num1 + num2); break;
                    case "-": s.push(num2 - num1); break;
                    case "*": s.push(num2 * num1); break;
                    case "/": s.push(num2 / num1); break;
                }
            }
        }
        return s.pop();
    }
}

public int longestPalindromeSubseq(String s){
    int[][] dp = new int[s.length()][s.length()];

    for(int i = s.length()-1; i >= 0; i--){
        dp[i][i] = 1;
        for(int j = i+1; j < s.length(); i++){
            if(s.charAt(i) == s.charAt(j)){
                dp[i][j] = d[i+1][j-1] + 2;//bbab, if bab, then no bbab, 
                                   //so they can not exist at the same time
                                   // The value of them are same!

            }else{
                dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
            }
        }
    }
    return dp[0][s.length() -1];
}

public String longestPalindrome(String s){
    if(s == null || s.length() <= 1){
        return s;
    }
    int len = s.length();
    int maxLen = 1;
    boolean[][] dp = new boolean[len][len];
    String res = null;
    for(int l = 0; l < len; l++){
        for(int i = 0; i < len - l; i++){
            int j = i + l;
            if(s.charAt(i) == s.charAt(j) && (j-i <=2 || dp[i+1][j-1])){
                dp[i][j] = true;
                if(j-i + 1 > maxLen){
                    maxLen = j-i + 1;
                    res = s.substring(i, j+1);
                }
            }
        }

    }
}

public String longestPalindrome(String s){
    if(s.isEmpty()){
        return null;
    }
    if(s.length() == 1){
        return s;
    }
    String longest = s.substring(0,1);
    for(int i = 0; i < s.length()-1; i++){
        String tmp = helper(s, i, i);
        if(tmp.length() > longest.length()){
            longest = tmp;
        }
        tmp = helper(s, i, i + 1);
        if(tmp.length() > longest.length()){
            longest = tmp;
        }
    }
    return longest;
}
private String help(String s, int i, int j){
    while(i>=0 && j <s.length() && s.charAt(i) == s.charAt(j)){
        i--;
        j++;
    }
    return s.substring(i+1, j);
}
public String findLongestWord(String s, List<String> d) {
        String longest = "";
        for(String dic : d){
            int i = 0;
            for(char c : s.toCharArray()){
                if(i<dic.length() && c == dic.charAt(i)) i++;
            }
            if( i == dic.length() && dic.length() >= longest.length()){
                if(dic.length() > longest.length() || dic.compareTo(longest) < 0) {
                    longest = dic;
                }
            }
        }
        return longest;
    }
//word break

public class Solution{
    public boolean wordBreak(String s, Set<String> dict){
            return help(s, dict, 0);
    }

    public boolean help(String s, Set<String> dict, int start){
        if (start == s.length()) return true;
        for(String d: dict){
            int len = d.length();
            int end = start + len;
            if(end > s.length()) continue;
            if(s.substring(start, end).equals(d)){
                if(help(s, dict, end)) {
                    return true;
                }
            }
        }
        return false;
    }
}

public class Solution{
    public boolean wordBreak(String s, Set<String> dict){
        boolean[] dp = new boolean[s.length()+1];
        dp[0] = true;
        for(int i = 0; i < s.length(); i++){
            if(!dp[i]) continue;
            for(String d: dict){
                int len = d.length();
                int end = i + len;
                if(end > s.length() || dp[end]) continue;
                if(s.substring(i, end)){
                    dp[end] = true;
                }
            }
        }
        return t[s.length()];
    }
}

public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] dp = new boolean[s.length()+1];
        dp[0] = true;
        for(int i = 0; i < s.length(); i++){
            if(dp[i]){
                for(int j = i+1; j < s.length()+1; j++){
                    String tomatch = s.substring(i, j);
                    if(wordDict.contains(tomatch)){
                        dp[j] = true;
                    }
                }
            }
            }
            return dp[s.length()];
        
    }




public void wiggleSort(int[] nums){
    int median = findKth(nums, nums.length/2);
    int n = nums.length;

    int left = 0;
    int right = n -1;
    int i = 0;
    while( i <= right){
        if(nums[newIndex(i, n)] < median){
            swap(nums, newIndex(i, n), newIndex(right, n));
            right--;


        }else if( nums[newIndex(i, n)] > median){
            swap(nums, newIndex(i, n), newIndex(left, n));
            left++;
            i++;

        }else{
            i++;
        }
    }
}


private int findKth(int[] nums, int k){
    int lo = 0;
    int hi = nums.length - 1;
    while(lo < hi){
        int m = partition(nums, lo, hi);
        if(m < k){
            lo = m + 1;
        }
        if( m > k){
            hi = m -1;
        }else break;
    }
    return nums[k];
    }





private int partition(int[] nums, int lo, int hi){
        int j = lo - 1;
        int pivot = nums[hi];

        for(int i = lo; i <= high -1 ; i++){
            if(nums[i] <= pivot ){
                swap(nums, ++j, i);
            }

        }
        swap(nums, j+1, hi);
        return nums[j+1];


}

private void swap(int[] a, int i, int j){
    int tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
}


private int newIndex(int index, int n){
    return (1 + 2*index)%(n | 1);
}

public int lengthOfLongestSubstring(String s){
    if(s.length() == 0) return 0;
    Map<Character, Integer> map = new HashMap<>();
    int max = 0;
    int j = 0;
    for(int i = 0; i < s.length(); i++){
        if(map.containsKey(s.charAt(i))){
            j = Math.max(j, map.get(s.charAt(i)));

        }
        map.put(s.charAt(i), i);
        max = Math.max(max, i - j + 1);
    }
}


public int devide(int dividend, int divisor){
    if(divisor == 0 || (dividend == Integer.MIN_VALUE && divisor == -1)) return Integer.MAX_VALUE;
    int res = 1;
    int sign = (dividend < 0) ^ (divisor < 0) ? -1 : 1;
    long dvd = Math.abs((long) dividend);
    long dvs = Math.abs((long) divisor);
    while(dvs <= dvd) {
        long tmp = dvs, mul = 1;
        while(dvd >= tmp<< 1){
            tmp <<= 1; mul << = 1;
        }
        dvd = dvd - tmp;
        res += mul;
    }
    return sing == 1? res : -res;
}

public int[] searchRange(int[] nums, int target){
    return helper(nums, target, 0, nums.length -1);
}
private int[] helper(int[] nums, int target, int lo, int hi){
    if(nums[lo] == target && nums[hi] == target) return new int[]{lo, hi};
    if(nums[lo] <= target && nums[hi] >=target){
        int mid = lo + (hi - lo)/2;
        int[] left = helper(nums, lo, mid);
        int[] right = helper(nums, mid, hi);
        if(left[0] == -1) return right;
        if(right[0] == -1) return left;
        return new int[]{left[0], right[1]};
    }
    return new int[]{-1, -1};
}

public int[] searchRange(int[] nums, int garget){
    int n = nums.length;
    int l = left(nums, target);
    if(l < 0 || l >= n || nums[l] != target) return new int[]{-1, -1};
    return new int[]{l, left(nums, target + 1) -1};
}
private int left(int[] nums, target){
    int lo = 0;
    int hi = nums.length -1;
    while(lo < hi){
        int mid = lo + (hi - lo)>>1;
        if(A[mid] < target){
            lo = mid + 1;
        }else{
            hi = mid;
        }
    }
    return lo;
}


vector<int> searchRange(int A[], int n, int target){
    int i = 0, j = n-1;
    vector<int> res(2, -1);
    while( i < j){
        int mid = i + (j -i)/2;
        if(A[mid] < target) i = mid + 1;
        else j = mid;
    }
    if (A[i] != target) return res;
    else res[0] = i; 
    j = n-1;
    while( i < j){
        int mid = i + (j -i + 1)/2;
        if(A[mid] > target) j = mid-1;
        else i = mid;
    }
    res[1] = j;
    return res;
}

public boolean exist(char[][] board, String word) {
    
    for(int i = 0; i < board.length; i++){
        for(int j = 0; j < board[0].length; j++){
            if(exist(board, i, j, word, 0)){
                return true;
            }
        }
    }
     return false;

}
private boolean exist(char[][] board, int i, int j, String word, int index){
    if(index == word.length()) return true;
    if(i < 0 || j < 0 || i >= board.length || j >= board[i].length) return false;
    if(board[i][j] != word.charAt(index)) return false;
    board[i][j] = 0;
    boolean match = exist(board, i+1, j, word, index+1)|| exist(board, i, j+1, word, index+1)|| exist(board, i-1, j,word,  index+1)|| exist(board, i, j-1, word, index +1);
    board[i][j] = word.charAt(index);
    return match;
}



public List<String> summaryRanges(int[] nums){
    List<String> res = new ArrayList<>();
    if(nums == null || nums.length ==0){
        return res;
    }
    for(int i = 0; i < nums.length;){
        int start = nums[i];
        while( i+1 < nums.length && nums[i+1] - nums[i] == 1){
            i++;
        }
        if(nums[i] !=  start){
            res.add(start+"->"+nums[i]);
        }
        else res.add("" + nums[i]);
        i++;
    }
    return res;
}




















