public ListNode rotateRight(ListNode head, int k) {
        if (head == null || k < 1) return head;
        ListNode point = head;
        int n = 1;
        while(point.next != null){
            point = point.next;
            n++;
        }
        point.next = head;
        k = n - k%n;
        while(k>0){
            point = point.next;
            k--;
        }
        head = point.next;
        point.next = null;
        return head;           
    }

 public int numberOfArithmeticSlices(int[] A) {
        if(A.length < 3 || A == null) return 0;
        int cur = 0, sum = 0;
        for(int i = 2; i< A.length; i++) {
            if((A[i-1] -A[i-2]) == (A[i] - A[i-1])){
                cur += 1;
                sum += cur;
            }
            else cur = 0;
        }
        return sum;
    }
public int[][] reconstructQueue(int[][] people){
    Arrays.sort(people, new Comparator<int[]>(){
        @override
        public int compare(int[] o1, int[] o2){
            return o1[0] != o2[0]? -o1[0]+o2[0]: o1[1]-o2[1];
        }
    });
    List<int[]> res =new LinkedList<>();
    for(int[] cur: people){
        res.add(cur[1], cur);
    }
    return res.toArray(new int[people.length][]);
}
 public String frequencySort(String s) {
        if(s == null) return null;
        if(s.length() == 0) return "";
        Map<Character, Integer> map = new HashMap<>();
        int max = 0;
        for(int i =0; i<s.length(); i++){
            map.put(s.charAt(i), map.getOrDefault(s.charAt(i),0) + 1);
            max = Math.max(max, map.get(s.charAt(i)));
        }
        List[] list = new List[max+1];
        for(Character c: map.keySet()) {
            if (list[map.get(c)] == null){
                list[map.get(c)] = new ArrayList<Character>();
            }
            list[map.get(c)].add(c);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = list.length - 1; i >0; i--){
            List<Character> list1 = list[i];
            if (list1 != null){
                for(Character c: list1){
                    for(int j = 0; j< i; j++){
                             sb.append(c);
            }
        }
    }
}
return sb.toString();
}
public String frequencySort(String s) {
    int[] count = new int[256];
    StringBuilder sb = new StringBuilder();
    List<List<Integer>> list = new ArrayList<>();
    int max = 0;
    for(int i = 0; i< s.length(); i++) {
        count[s.charAt(i)]]++;
        max = Math.max(max, count[s.charAt[i]);
    }
    for(int i = 0; i< max+1; i++) list.add(new ArrayList<>());
    for(int i = 0; i< count.length; i++) if(count[i] != 0) list.get(count[i]).add(i);
    for(int i = list.size() -1; i>=0; i--) {
        if(list.get(i) != null){
            List<Integer> tmp = list.get(i);
            for(Int c : tmp){
                for(int j = 0; j<i; j++){
                    sb.append(Character.toChars(c));
                }
            }
        }
    }
    return sb.toString();

}
public List<Integer> findDuplicates(int[] nums) {
    List<Integer> res = new ArrayList();
    for(int num : nums) {
        int n = Math.abs(num);
        int index = n - 1;
        if(nums[index] < 0) res.add(n);
        else nums[index] = - nums[index];
    }
    for(int i=0; i<nums.length; i++){
        nums[i] = Math.abs(nums[i]);
    }
    return res;
}

public List<List<Integer>> subsets(int[] S){
    Arrays.sort(S);
    int totalNumber = 1 << S.length;
    List<List<Integer>> collection = new ArrayList<List<Integer>>(totalNumber);
    for( int i =0; i < totalNumber; i++){
        List<Integer> set = new ArrayList<Integer>();
        for(int j = 0; j<S.length; j++){
            if((i & (1 << j))) != 0) {
                set.add(S[j]);
}
        }
        collections.add(set);
    }
    return collections;

}

public List<List<Integer>> subsets(int[] S){
    List<List<Integer>> result = new ArrayList<List<Integer>>();
    if(S == null || S.length ==0) return result;
    ArrayList<Integer> list = new ArrayList<Integer>();
    Arrays.sort(S);
    subsetsHelper(result, list, S, 0);
    return result;
}
private void subsetsHelper(List<List<Integer>> result, ArrayList<Integer> list, int[] num, int pos){
    result.add(new ArrayList<>(list));
    for(int i = pos; i< num.length; i++){
        list.add(num[i]);
        subsetsHelper(result, list, num, i+1);
        list.remove(list.size() - 1);
    }
}
 public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (nums == null && nums.length ==0) return result;
        Arrays.sort(nums);
        List<Integer> list = new ArrayList<Integer>();
        subsethelp(result, list, nums, 0);
        return result;
        
    }
    private void subsethelp(List<List<Integer>> result, List<Integer> list, int[] nums, int pos){
        result.add(new ArrayList<Integer>(list));
        for(int i = pos; i<nums.length; i++){
            if(i > pos && nums[i] == nums[i-1]) continue;
            list.add(nums[i]);
            subsethelp(result, list, nums, i+1);
            list.remove(list.size() - 1);
    }
}
 public List<List<Integer>> subsetsWithDup(int[] nums) {
       Arrays.sort(nums);
       int len = nums.length;
       int max = 1<< nums.length;
       List<List<Integer>> res = new ArrayList<List<Integer>>();
       for(int i = 0; i< max; i++){
           List<Integer> list = new ArrayList<Integer>();
           boolean duplicate = false;
           for (int j = 0; j<len; j++){
               if((i& (1<<j) ) !=0) {
                   if(j>0 && nums[j] == nums[j-1] && (i & 1<<(j-1)) == 0){
                       duplicate = true;
                       break;
                   }
                 
                list.add(nums[j]);
                   
                   
               }
           }
           if(duplicate == false){
           res.add(list);
       }
       }
     return res;  
}
public List<List<Integer>> permute(int[] num) {
    List<List<Integer>> ans ＝ new ArrayList<List<Integer>>();
    if(num.length() == 0) return res;
    List<Integer> list = new ArrayList<>();
    list.add(num[0]);
    ans.add(list);
    for(i = 1; i< num.length; i++){
        List<List<Integer>> new_ans = new ArrayList<List<Integer>>();
        for(j = 0; j<=i; j++) {
            for(List<Integer> l : ans){
                List<Integer> new_l = new ArrayList<Integer>(l);
                new_l.add(j, num[i]);
                new_ans.add(new_l);
            }
        }
        ans = new_ans;
    }
    return ans;
}

public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        List<Integer> list = new ArrayList<Integer>();
        Helper(result, list, nums);
        return result;
        
    }
    private void Helper(List<List<Integer>> result, List<Integer> list, int[] nums){
        if(list.size() == nums.length) {
            result.add(new ArrayList<Integer>(list)); 
            return;
    }
    else{
    for(int i = 0; i< nums.length; i++){
        if (list.contains(nums[i])) continue;
        list.add(nums[i]);
        Helper(result, list, nums);
        list.remove(list.size() - 1);
    }
}
}
public List<List<Integer>> permute(int[] num){
    List<List<Integer>> result = new ArrayList<List<Integer>>();
    permute(num,0,result);
    return result;
}
public void permute(int[] num, int begin, List<List<Integer>> result){
    if(begin >= num.length){
        List<Integer> list = new ArrayList<Integer>();
        for(int i = 0; i < num.length; i++){
            list.add(num[i]);
        }
        result.add(list);
        return;
    }
    for(int i = begin; i< num.length; i++){
        swap(begin,i, num);
        permute(num, begin+1, result);
        swap(begin,i,num);
    }

}
public void swap (int x, int y, int[] num){
    int temp = num[x];
    num[x] = num[y];
    num[y] = temp;
}

public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if(nums== null) return result;
        if(nums.length ==0) {
            result.add(new ArrayList<Integer>());
            return result;
        }
        Arrays.sort(nums);
        List<Integer> list = new ArrayList<Integer>();
        list.add(nums[0]);
        result.add(list);
        for(int i = 1; i < nums.length; i++){
            Set<ArrayList<Integer>> currentset = new HashSet<ArrayList<Integer>>();
            for(List<Integer> l : result){
            for(int j = 0; j<= i; j++){
                    ArrayList<Integer> nlist = new ArrayList<Integer>(l);
                    nlist.add(j, nums[i]);
                    currentset.add(nlist);
                }
            }
            result = new ArrayList<List<Integer>>(currentset);
        }
        return result;
    }


public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(nums);
    List<Integer> list = new ArrayList<>();
    backrack(res,list, nums, new boolean[nums.length]);
    return res;
}

private void backrack(List<List<Integer>> res, List<Integer> list, int[] nums, boolean[] used){
    if(list.size() == nums.length){
        res.add(new ArrayList<>(list));
    }else{
        for(int i =0; i<nums.length; i++){
            if(used[i]) continue;
            if(i > 0 && nums[i] == nums[i-1] && !used[i-1]) continue;
            used[i] = true;
            list.add(nums[i]);
            backrack(res, list, nums, used);
            used[i] = false;
            list.remove(list.size() - 1);
        }
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
                list.add(candidates[i]);
                Help(res, list,candidates, target-candidates[i], i);
                list.remove(list.size() - 1);
            }
        }
    }
     public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> list = new ArrayList<>();
        Arrays.sort(candidates);
        Help(res, list, candidates, target, 0);
        return res;
        }
    private void Help(List<List<Integer>> res, List<Integer> list, int[] candidates, int target, int start){
        if (target < 0) return;
        if (target == 0){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        else{
            for(int i = start; i< candidates.length; i++){
                if(i>start && candidates[i] == candidates[i-1]) continue;
                list.add(candidates[i]);
                Help(res, list,candidates, target - candidates[i], i+1);
                list.remove(list.size() - 1);
            }
            
        }
    }
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<List<String>>();
        List<String> list = new ArrayList<String>();
        help(res, list, s, 0);
        return res;
        
    }
    private void help(List<List<String>> res, List<String> list, String s, int start){
        if(start == s.length()) res.add(new ArrayList<String>(list));
        else{
            for(int i = start; i<s.length(); i++){
                if (isPalindrome(s, start, i)){
                list.add(s.substring(start, i+1));
                help(res, list, s, i+1);
                list.remove(list.size() -1 );
            }
        }
    }
    }
    public boolean isPalindrome(String sub,int start, int end){
        int i = start;
        int j = end;
        while(i<j){
            if(sub.charAt(i) != sub.charAt(j)) return false;
            i++;
            j--;
    }
    return true;
}
public int minCut(String s){
    int n = s.length();
    int[] cut = new int[n+1];
    char[] c = s.toCharArray();
    for (int i = 0; i<=n; i++) cut[i] = i-1;
    for(int i = 0; i<n; i++){
        for(int j = 0; i-j>=0 && i+j<n && c[i-j] ==c[i+j]){
            cut[i+j+1] = Math.min(cut[i+j+1], 1 + cut[i-j]);

        }
        for(int j = 1; i-j+1 >=0 && i+j < n && c[i-j+1] == c[i+j]){
            cut[i+j+1] = Math.min(cut[i+j+1], 1 + cut[i-j+1]);
        }
        return cut[n];
    }
}
public int minCut(String s) {
    int n = s.length();
 
    boolean[][] dp = new boolean[n][n];
    int cut[] = new int[n];
    
    for (int j = 0; j < n; j++) {
        cut[j] = j; //set maximum # of cut
        for (int i = 0; i <= j; i++) {
            if (s.charAt(i) == s.charAt(j) && (j - i <= 1 || dp[i+1][j-1])) {
                dp[i][j] = true;
 
                // if need to cut, add 1 to the previous cut[i-1]
                if (i > 0){
                    cut[j] = Math.min(cut[j], cut[i-1] + 1);
                }else{
                // if [0...j] is palindrome, no need to cut    
                    cut[j] = 0; 
                }   
            }
        }
    }
 
    return cut[n-1];
}

public int magicalString(int n) {
    if(n <=0) return 0;
    if(n <=3) return 1;
    int[] a = new int[n+1];
    a[0] = 1; a[1] = 2; a[2] = 2;
    int head = 2; tail = 3, num = 1, result =1;
    while(tail < n) {
        for(int i = 0; i <a[head]; i++){
            a[tail] = num;
            if(num == 1 && tail < n) result++;
            tail++;
        }
        num = num^3;
        head++
    }
    return result;
}


public List<Integer> topKFrequent(int[] nums, int k){
    Map<Integer, Integer> map = new HashMap<>();
    for(int n: nums){
        map.put(n, map.getOrDefault(n,0) + 1);
    }
    PriorityQueue<Map.Entry<Integer,Integer>> maxHeap = new PriorityQueue<>((a,b)->(b.getValue() - a.getValue()));
    for(Map.Entry<Integer, Integer> entry: map.entrySet()){
        maxHeap.add(entry);
    }
    List<Integer> res = new ArrayList<>();
    while(res.size()<k){
        Map.Entry<Integer,Integer> entry = maxHeap.poll();
        res.add(entry.getKey());
    }
    return res;
}

 public List<Integer> topKFrequent(int[] nums, int k) {
        List<Integer> res = new ArrayList<Integer>();
        Map<Integer, Integer> map = new HashMap<Integer,Integer>();
        for(int i = 0; i< nums.length; i++){
            map.put(nums[i], map.getOrDefault(nums[i],0)+1);
        }
        List<Integer>[] list = new List[nums.length + 1];
        for(int key: map.keySet()) {
            int freq = map.get(key);
            if (list[freq] == null) {
                list[freq] = new LinkedList<Integer>();
            }
            list[freq].add(key);
        }
        for (int j = list.length -1; j >=0; j--){
            if(list[j] != null){
                for(int i = 0; i<list[j].size() && res.size()<k; i++){
                    res.add(list[j].get(i));
                }
            }
        }
        return res;
    }
public class Solution {
    private ListNode head;
    private Random random;

    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    public Solution(ListNode head) {
        this.head = head;
        this.random = new Random();
        
    }
    
    /** Returns a random node's value. */
    public int getRandom() {
        int res = 0;
        ListNode point = head;
        int count = 1;
        while(point!=null){
            if(random.nextInt(count) == 0) res = point.val;
            point = point.next;
            count++;
        }
      return res;  
    }
}

public String multiply(String num1, String num2) {
        int m = num1.length();
        int n = num2.length();
        int[] res = new int[m+n];
        for(int i = m-1; i>=0; i--){
            for (int j = n-1; j>=0; j--){
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int p1 = i+j, p2 = i+j+1;
                int sum = mul + res[p2];
                res[p2] = sum % 10;
                res[p1] = sum / 10;
            }
            
        }
        StringBuilder sb = new StringBuilder();
        for(int num: res){ if(!(sb.length() == 0 && num ==0)) sb.append(num);}
        
        return  sb.length() == 0 ? "0": sb.toString();
        
    }
    public String multiply(String num1, String num2) {
        int m = num1.length();
        int n = num2.length();
        int[] res = new int[m+n];
        for(int i = m-1; i>=0; i--){
            for (int j = n-1; j>=0; j--){
                int sum = 0;
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int p1 = i+j, p2 = i+j+1;
                sum = mul + res[p2];
                res[p2] = sum % 10;
                res[p1] += sum / 10; // pay attention here, need += since this position may have value in the last round.
            }
            
        }
        StringBuilder sb = new StringBuilder();
        for(int num: res){ if(!(sb.length() == 0 && num ==0)) sb.append(num);}
        
        return  sb.length() == 0 ? "0": sb.toString();
        
    }


public int bulbSwitch(int n){
    int count = 0;
    for(i = 1; i<=n; i++){
        in switchi = help(i);
        if(switchi % 2 == 1) count++
    }
return count;
}
public int help(int n){
    int c = 0;
    for(int i = 1;i<=n; i++){
        if(n%i == 0) c++;
    }
    return c;
}
public class Solution{
    TreeNode pre = null;
    public boolean isValidBST(TreeNode root){
        if (root == null) return true;
        if (!isValidBST(root.left)) return false;
        if(pre !=null && pre.val >= root.val) return false;
        pre = root;
        return isValidBST(root.right);
    }
}

public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        if (root == null) return res;
        
        TreeNode cur, pre;
        cur = root;
        while(cur != null){
        if(cur.left == null) {
            res.add(cur.val);
            cur = cur.right;
        }
        else{
            pre = cur.left;
            while(pre.right != null && pre.right != cur){
                pre = pre.right;
            }
            if(pre.right == null){
                pre.right = cur;
                cur = cur.left;
                
            }
            else{
                pre.right = null;
                res.add(cur.val);
                cur = cur.right;
            }
        }
        
    }
    return res;
    }


public List<Integer> inorderTraversal(TreeNode root){
    List<Integer> list = new ArrayList<Integer>();
    Stack<TreeNode> stack = new Stack<TreeNode>();
    TreeNode cur = root;
    while(cur != null || !stack.empty()){
        while(cur != null){
            stack.add(cur);
            cur = cur.left;
        }
        cur = stack.pop();
        list.add(cur.val);
        cur = cur.right;
    }
    return list;
}
public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        if (root == null) return res;
        TreeNode cur = root;
        TreeNode left;
        while( cur != null) {
            
            if(cur.left == null){
                res.add(cur.val);
                cur = cur.right;
            }
            else{
                left = cur.left;
                while(left.right != null && left.right != cur){
                    left = left.right;
                }
                if(left.right ==null){
                    res.add(cur.val);
                    left.right = cur.right;// this line can also be left.right = cur;
                    cur = cur.left;
                }
                else{
                    left.right = null;
                    cur = cur.right;
                }
                
            }
        }
        return res;
    }
public int[] findDiagonalOrder(int[][] matrix) {
    if (matrix == null && matrix.length == 0) return new int[0];
    int m = matrix.length;
    int n = matrix[0].length;
    int[] arr = new int[m*n];
    int row = 0, col = 0, d=0;
    int[][] direction = [[-1,1],[1,-1]];
    for(int i=0; i < m*n; i++){
        result[i] = matrix[row][col];
        row += direction[d][0];
        col += direction[d][1];
        if(row >= m) {row = m-1; col += 2; d = 1-d;}
        if(col >= n) {col = n-1; row += 2; d = 1-d;}
        if(row < 0) {row = 0; d = 1 - d;}
        if(col < 0) {col = 0; d = 1 - d;}
    }
    return result;
    
}
public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        Stack<TreeNode> stack = new Stack();
        if (root == null) return res;
        stack.push(root);
        while(!stack.isEmpty()) {
                  TreeNode cur = stack.pop();
                  res.add(cur.val);
                  if(cur.right != null) stack.push(cur.right);
                  if(cur.left != null) stack.push(cur.left);
        }
        return res;
    }
public class Solution {
    int max = 0;
    public int[] findFrequentTreeSum(TreeNode root) {
        int[] res = new int[0];
        if(root == null) return res;
        List<Integer> list = new ArrayList<>();
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        help(root, map);
        for(int i : map.keySet()){
            if(map.get(i)==max) list.add(i);
        }
        return list.stream().mapToInt(i->i).toArray();
        
    }
    private int help(TreeNode root, Map<Integer,Integer> map){
       if(root == null) return 0;
       int total = root.val + help(root.left, map) + help(root.right,map);
       int count = map.getOrDefault(total, 0) + 1;
       max = Math.max(max, count);
       map.put(total, count);
       return total;
}
}
public boolean canPartition(int[] nums) {
        if(nums == null || nums.length ==0) return true;
        int sum = 0;
        for(int num: nums){
            sum += num;
        }
        if((sum & 1) == 1) return false;
        sum /= 2;
        boolean dp[] = new boolean[sum + 1];
        dp[0] = true;
        for(int i = 0; i< nums.length; i++){
            for(int j = sum; j>=nums[i]; j--){
                dp[j] = dp[j-nums[i]] || dp[j];
            }
        }
        return dp[sum];
public int findTargetSumWays(int[] nums, int S) {
       if(nums == null || nums.length ==0) return 0;
       int sum = 0;
       for(int n: nums){
           sum += n;
       }
       if(sum < Math.abs(S)) return 0;
       if((sum + S) % 2 ==1) return 0;
       int dp[] = new int[(sum+S)/2 + 1];
       dp[0] = 1;
       for(int i = 0; i< nums.length; i++){
           for(int j = (sum+S)/2; j>=nums[i]; j--){
               dp[j] = dp[j] + dp [j- nums[i]];
           } 
       }
    return dp[(sum+S)/2];
    }

public class Solution{
    int result = 0;
    public int findTargetSumWays(int[] nums,int S){
        if(nums == null || nums.length ==0) return result;
        int n = nums.length;
        int[] sums = new int[n];
        sums[n-1] = nums[n-1];
        for(int i = n-2; i>=0; i--){
            sums[i] = sums[i+1] + nums[i];
        }
        helper(nums, sums, S, 0);
    }
    public void helper(int[] nums, int[] sums, int target, int pos){
        if(target ==0 && pos == nums.length -1) {
            result++;
            return;
        }
        if(sums[pos] < Math.abs(target)) return;
        helper(nums, sums, target + nums[i], pos+1);
        helper(nums, sums, target - nums[i], pos+1);
}

 public ListNode partition(ListNode head, int x) {
        ListNode dummy = new ListNode(0);
       
        dummy.next = head;
         ListNode p1 = dummy;
        ListNode large = new ListNode(0);
        ListNode p2 = large;
        while(p1.next!= null){
            if(p1.next.val >= x){
                p2.next = p1.next;
                p2= p2.next;
                p1.next = p1.next.next;
            }
            else p1 = p1.next;
        }
        p2.next = null;
        p1.next = large.next;
        return dummy.next;
    }

 public int totalHammingDistance(int[] nums) {
        if(nums ==null || nums.length ==0) return 0;
        int res =0;
        for(int i = 0; i < 32; i++){
            int count = 0;
            for(int num : nums){
                if((num & (1<<i)) != 0) count++;
            }
           res += (nums.length-count)*count; 
        }
        return res;
    }
public List<String> findRepeatedDnaSequences(String s) {
        Set<String> once = new HashSet<>();
        Set<String> more = new HashSet<>();
        List<String> res = new ArrayList<String>();
        if(s==null || s.length() ==0) return res;
        for(int i = 0 ; i< s.length() -9; i++){
            String sub = s.substring(i, i+10);
            if(!once.add(sub)){
                more.add(sub);
            }
        }
       res = new ArrayList(more);
       return res;
    }

public int maxProduct(String[] words){
    int n = words.length;
    int[] elements = new int[n];
    for(int i =0; i<n;i++){
        for(int j = 0; j< words[i].length(); j++){
            elements[i] |= 1 << (words[i].charAt(j) - 'a');
        }
    }
    int ans = 0;
    for(int i = 0; i < n; i++){
        for(int j = i+1; j < n; j++){
            if((elements[i] & elements[j]) == 0)
                ans = Math.max(ans, words[i].length() * words[j].length());
        }
    }
    return ans;
}
 public int maxProduct(String[] words) {
        int n = words.length;
        int[] element = new int[n];
        for(int i = 0; i< n; i++){
            for(int j = 0; j< words[i].length(); j++){
              element[i] |= 1 << (words[i].charAt(j) - 'a');  
            }
         
        }
        int res = 0;
        for(int i = 0; i< n-1; i++){
            for(int j = i+1; j<n; j++){
                if((element[i] & element[j]) ==0)
                res = Math.max(res, words[i].length()*words[j].length());
            }
        }
        return res;
        
    }
public int countNumbersWithUniqueDigits(int n) {
        if(n ==0) return 1;
        int res = 10;
        int first = 9;
        int second = 9;
        for(int i = n; i>1 && second >0; i--){
            first = first*second;
            res += first;
            second = second - 1;
        }
        return res;
        
    }
public int combinationSum4(int[] nums, int target){
    int[] comb = new int[target+1];
    comb[0] = 0;
    for(int i =1; i< comb.length; i++){
        for(int j = 0; j<nums.length; j++){
            if (i- nums[j] >= 0){
                comb[i] = comb[i] + comb[i-nums[j]];
            }
        }
    }
    return comb[target];
}

public class Solution {
   private int[] dp;
   public int combinationSum4(int[] nums, int target) {
    dp = new int[target + 1];
    Arrays.fill(dp, -1);
    dp[0] = 1;
    help(nums, target);
    return dp[target];
}
public int help(int[] nums, int target){
    if(dp[target] != -1){
       return dp[target];
    }
    int res= 0;
    for(int num:nums){
        if(target >= num){
            res += help(nums, target - num);
        }
    }
    dp[target] = res;
    return dp[target];

}
    
}
public int coinChange(int[] coins, int amount){
    if(amount ==0) return 0;
    if(coins == null || coins.length ==0) return -1;
    int[] dp = new int[amount+1];
    Arrays.fill(dp, amount+1);
    dp[0] = 0;
    for(int i = 1; i<=amount; i++){
        for(int j=0; j<coins.length; j++){
            if(i>coins[j]){
                dp[i] = Math.min(dp[i], dp[i-coins[j]] + 1);
            }
        }

    }
    return dp[amount]>amount? -1: dp[amount];
}
//Unique binary search tree:
 public int numTrees(int n) {
        int[] G = new int[n+1];
        G[0] = 1;
        G[1] = 1;
        for(int i = 2; i<=n; i++){
            for(int j = 1;j<=i;j++){
                G[i] += G[j-1]*G[i-j];
            }
        }
        return G[n];
    }
  public List<TreeNode> generateTrees(int n) {
    if (n==0) return new ArrayList<TreeNode>();
    return genTrees(1,n);
}
private List<TreeNode> genTrees(int start, int end){
    List<TreeNode> list = new ArrayList<TreeNode>();
    if(start > end){
        list.add(null);
        return list;
    }
    if(start == end){
        list.add(new TreeNode(start));
        return list;
    }
    for(int i = start; i<=end; i++){
        List<TreeNode> left = genTrees(start, i-1);
        List<TreeNode> right = genTrees(i+1, end);
        for(TreeNode lnode : left){
            for(TreeNode rnode:right){
                TreeNode root = new TreeNode(i);
                list.add(root);
                root.left = lnode;
                root.right = rnode;
            }
        }
    }
    return list;
}

public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k){
    List<int[]> res = new ArrayList<int[]>();
    if(nums1.length ==0 || nums2.length ==0|| k ==0) return res;
    PriorityQueue<int[]> queue = new PriorityQueue<>((a,b)-> (a[0] +a[1]) - (b[0]+b[1]));
    for(int i = 0; i<Math.min(nums1.length, k); i++){
        queue.offer(new int[]{nums1[i], nums2[0],0});
    }
    while(k-->0 && !queue.isEmpty()){
        int[] cur = queue.poll();
        res.add({cur[0],cur[1]});
        if(cur[2] == nums2.length -1) continue;
        int nextfirst = cur[0];
        
        int nextlevel = cur[2]+1;
        int nextsecond = cur[nextlevel];
        queue.offer(new int[]{nextfirst, nextsecond, nextlevel})
    }
    return res;
}

public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
   int m = nums1.length;
   int n = nums2.length;
   List<int[]> list = new ArrayList<>();
   if(m==0 || n==0 || k < 1) return list;
   PriorityQueue<Data> queue = new PriorityQueue<>((a,b) ->(a.val - b.val));
   queue.offer(new Data(0,0, nums1[0] + nums2[0]));
   while(!queue.isEmpty() & k > 0){
    Data d = queue.poll();
    int[] pair = (nums1[d.i], nums2[d.j]);
    list.add(pair);
    k--;
    if(d.j<n-1) queue.offer(new Data(d.i,d.j+1, nums1[d.i] + nums2[d.j + 1]));
    if(d.j == 0 && d.i < m-1) queue.offer(new Data(d.i + 1, 0, nums1[d.i+1] + nums2[0]));
   }
   return list;

}
class Data{
    int i;
    int j;
    int val;
    public Data(int i, int j, int val){
        this.i = i;
        this.j = j;
        this.val = val;
    }
}
public void connect(TreeLinkNode root) {
        if(root == null) return;
        if(root != null && root.left ==null) {
            root.next = null;
            return;
        }
        while(root != null && root.left != null) {
            TreeLinkNode p = root;
            while(p != null){
                p.left.next = p.right;
                p.right.next = p.next == null? null: p.next.left;
                p = p.next;
                
            }
            root = root.left;
        }
        
    }
public void(connect TreeLinkNode root){
    if(root == null) return;
    if(root !=null && root.left ==null) {root.next =null; return;}
    while(root !=null){
        TreeLinkNode point = root;
        while(point != null && point.left !=null){
            point.left.next = point.right;
            point.right.next = point.next == null? null: point.next.left;
            point = point.next;
        }
        root = root.left;
    }
}
public class Solution {
    public void connect(TreeLinkNode root) {
        if(root == null) return;
        if(root != null && root.left ==null) {
            root.next = null;
            return;
        }
        help(root.left, root.right);
        
    }
    private void help(TreeLinkNode left, TreeLinkNode right){
        if(left == null)  return;
        left.next = right;
        help(left.left, left.right);
        help(left.right, right.left);
        help(right.left, right.right);
        right.next =null;
    }
}
 public void connect(TreeLinkNode root){
    TreeLinkNode head = root;
    TreeLinkNode prev = null;
    TreeLinkNode curr = null;
    while(head != null){
        curr = head;
        prev = null;
        head = null;
    
    while(curr != null){
        if(curr.left !=null){
            if(prev != null) 
                prev.next = curr.left;
            else 
                head = curr.left;
            prev = curr.left;
        }
        if(curr.right != null){
            if(prev !=null)
                prev.next = curr.right;
            else
                head = curr.right;
            prev = curr.right;
        }
        curr = curr.next;
    }

 }
 public void connect(TreeLinkNode root) {
        TreeLinkNode tempnode = new TreeLinkNode(0);
        while(root != null) {
            TreeLinkNode child = tempnode;
            TreeLinkNode p = root;
            while(p != null){
            if(p.left != null){
                child.next = p.left;
                child = child.next;
            }
            if(p.right != null){
                child.next = p.right;
                child = child.next;
            }
            p = p.next;
        }
        root = tempnode.next;
        tempnode.next = null;
        }
    }

public int[] nextGreaterElements(int[] nums){
    int[] res = new int[nums.length];
    Arrays.fill(res, -1);
    int n = nums.length;
    Stack<Integer> stack = new Stack();
    for(int i = 0; i < 2*n ; i++){
        int num = nums[i%n];
        while( !stack.isEmpty() && nums[stack.peek()] < num){
            res[stack.pop()] = num;
        }
        if(i<n) stack.push(i); 
    }
    return res;
}

public List<List<Integer>> subset(int[] S){
    
}


