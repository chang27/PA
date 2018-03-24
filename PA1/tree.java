public int maxDepth(TreeNode root) {
	if(root == null) return 0;
	Deque<TreeNode> stack = new LinkedList<TreeNode>();
	stack.push(root);
	int count = 0;
	while(!stack.isEmpty()) {
		int size = stack.size();
		while(size -- >0) {
			TreeNode cur = stack.pop();
			if(cur.left !=null)
				stack.addLast(cur.left);
			if(cur.right !=null)
				stack.addLast(cur.right);
		}
		count++;
	}
	return count;
}
public int minDepth(TreeNode root){
	if(root ==null) return 0;
	if (root.left == null || root.right == null) {
		int depth = 1+ Math.max(minDepth(root.right), minDepth(root.left));
	}
	else {depth = 1+ Math.min(minDepth(root.right), minDepth(root.left))}
	return depth;
}
putlic ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
	ArrayList result = new ArrayList();
	if(root == null) {
		return result;
	}
	Queue<TreeNode> queue = new LinkedList<TreeNode>();
	queue.offer(root);
	while(!queue.isEmpty()) {
		ArrayList<Integer> level = new ArrayList<Integer>();
		int size = queue.size();
		for(int i = 0; i<size; i++) {
			TreeNode head = queue.poll();
			level.add(head.val);
			if(head.left != null){
				queue.offer(head.left);
			}
			if(head.right != null){
				queue.offer(head.right);
			}

		}
		result.add(level);
	}
	return result;
}
public int addDigits(int num) {
        int sum = 0;
        if (num ==0) return 0;
        if ((num %9) ==0 ) sum = 9;
        else sum =(num%9);
        return sum;
        
    }

int digitSquareSum(int n) {
	int sum = 0,tmp;
	while(n) {
		tmp = n % 10;
		sum = sum + tmp*tmp;
		n /= 10;
	}
	return sum;

}
boolean isHappy (int n) {
	int slow, fast;
	slow = fast = n;
	do{
		slow = digitSquareSum(slow);
		fast = digitSquareSum(fast);
		fast = digitSquareSum(fast);
		if (fast == 1) return 1;

	} while(slow != fast);
	return 0;
}
public boolean isHappy(int n){
	int sum = 0;
	HashSet<Integer> hash = new HashSet<Integer>();
	while (n != 1) { 
	int t = 0;
	while(n!=0){
		t += (n % 10)*(n % 10);
		n /= 10;
	}
	n = t;
	if (hash.contains(n)) return false;
	else hash.add(n);
    }
    return true;
}
 public boolean isUgly(int num) {
        int[] divide = new int[] {2,3,5};
        if(num >0) {
            for(int i : divide) {
                while( num % i ==0 ){
                    num /= i;
                }
            }
        }
            
        if(num ==1) return true;
        else return false;
    }
public int countPrimes(int n) {
	boolean[] isPrime = new boolean[n];
	for(int i =2; i<n; i++) {
		isPrime[i] = true;
	}
	int count =0;
	for(int i = 2; i*i < n; i++) {
		if(!isPrime[i]) continue;
		for(int j = i*i; j<n; j += i){
			isPrime[j] = false;
		}
	}
	for(int i =2; i<n; i++){
		if (isPrime[i]) count++;
	}
    return count;
}
public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        Queue<TreeNode> stack = new LinkedList<TreeNode>();
        stack.offer(root);
        while(!stack.isEmpty()){
            TreeNode node = stack.poll();
            TreeNode left = node.left;
            node.left = node.right;
            node.right = left;
            if(node.left != null) stack.offer(node.left);
            if(node.right != null) stack.offer(node.right);
        }
        return root;
}
public class Solutions{
public TreeNode invertTree(TreeNode root) {
	if (root == null) return null;
	helper(root);
	return root;
}
	public void helper(TreeNode p) {
		TreeNode node = p.left;
		p.left = p.right;
		p.right = node;
		if(p.left != null) helper(p.left);
		if(p.right != null) helper(p.right);
	}
}

public void moveZeroes(int[] nums) {
        int real = 0;
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] != 0){
                nums[real] = nums[i];
                real++;
            }
        }
        for(int j= real; j<nums.length; j++){
            nums[j] = 0;
        }
        
}
 public int[] intersection(int[] nums1, int[] nums2) {

	    Set<Integer> set = new HashSet<>();
	    Set<Integer> result = new HashSet<>();
	    for(int i=0; i<nums1.length; i++) {
		    set.add(nums1[i]);
	   }
	    for(int i=0; i<nums2.length; i++) {
		    if(set.contains(nums2[i])) {
			    result.add(nums2[i]);
		    }
	    }
    	int[] finalresult = new int[result.size()];
	    int i =0;
	    for(Integer num: result) {
		    finalresult[i] = num;
		    i++;
	    }
	   return finalresult;
        
  }
  public int[] intersection(int[] nums1, int[] nums2) {
  	Set<Integer> set = new HashSet<>();
  	Array.sort(nums2);
  	for(Integer num : nums1) {
  		if(binarySearch(nums2, num)){
  			set.add(num);
  		}
  	}
  	int i = 0;
  	int[] result = new int[set.size()];
  	for (Integer num: set) {
  		result[i++] = num;
  	}
  	return result;
  }
  public boolean binarySearch(int[] nums, int target) {
  	int low = 0;
  	int high = nums.length - 1;
  	while( low <= high) {
  		int mid = (low + high) / 2;
  		if(nums[mid] == target) return true;
  		if(nums[mid] > target) high = mid -1;
  		if(nums[mid] < target) low = mid + 1;
  	}
  	return false;
  }

public int[] intersect(int[] nums1, int[] nums2) {
	HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
	ArrayList<Integer> result = new ArrayList<Integer>();
	for (int i=0; i<nums1.length; i++) {
		if (map.containsKey(nums1[i])) map.put(nums1[i]，map.get(nums1[i])+1);
		elese map.put(nums1[i], 1);
	}
	for(int i=0; i<nums2.length; i++){
		if(map.containsKey(nums2[i]) && map.get(nums2[i]) > 0){
			result.add(nums2[i]);
			map.put(nums2[i],map.get(nums2[i])-1);
		}
	}
	int[] finalresult = new int[result.size()];
	int i=0;
	for(Integer n : result){
		finalresult[i] = n;
		i++;
	}
	return finalresult;
}
public int[] intersect(int[] nums1, int[] nums2){
	Arrays.sort(nums1);
	Arrays.sort(nums2);
	ArrayList<Integer> list = new ArrayList<Integer>();
	int p1 = 0, p2 = 0;
	while(p1< nums1.length && p2 < nums2.length) {
		if(nums1[p1] < nums2[p2]) {
			p1++;
		}else if (nums1[p1] > nums2[p2]) {
			p2++
		}else{
			list.add(nums1[p1]);
			p1++;
			p2++;
		}
	}
	int[] result = new int[list.size()];
	int i=0;
	while(i<list.size()){
		result[i] = list.get(i);
		i++;
	}
	return result;
}
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
	if(l1==null) return l2;
	if (l2 == null) return l1;
	ListNode first = new ListNode(0);
	ListNode current = first;
	while(l1 !=null &&l2 != null){
		if (l1.val< l2.val){
			current.next = l1;
			l1 = l1.next;
		}else{
			current.next = l2;
			l2 = l2.next;
		}
		current = current.next;
	}
	if(l1 == null) current.next = l2;
	if(l2 ==null) current.next = l1;
	return first.next;
}
 public int minMoves2(int[] nums) {
        Arrays.sort(nums);
        int mid = (nums.length / 2);
        int count =0;
        for (int i = 0; i<nums.length; i++) {
            count += Math.abs(nums[i] - nums[mid]);
            
        }
        return count;
        
 }
 public boolean canConstruct(String ransomNote, String magazine) {
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        for(char c:magazine.toCharArray()){
            int count = map.getOrDefault(c,0) + 1;
            map.put(c, count);
        }
        for(char c: ransomNote.toCharArray()) {
            if(map.containsKey(c)) {
                int newc = map.get(c)-1;
                if(newc < 0) return false;
                map.put(c, newc);
            }
            else return false;
        }
        return true;

    }
 public int sumOfLeftLeaves(TreeNode root) {
        if (root ==null) return 0;
        int lef = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            TreeNode current = queue.poll();
            if(current.left != null){
                if(current.left.left == null && current.left.right ==null){
                lef += current.left.val;
            }else queue.offer(current.left);
            }
            if(current.right != null)  queue.offer(current.right);
            }
        
        return lef;
    }

   public int sumOfLeftLeaves(TreeNode root) {
        if (root ==null) return 0;
        int lef = 0;
        if(root.left !=null){
            if(root.left.left == null && root.left.right == null) lef += root.left.val + sumOfLeftLeaves(root.right);
            else lef += sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
        }
        else lef += sumOfLeftLeaves(root.right);
        return lef;
    }
    public int lengthOfLastWord(String s) {
        if(s ==null || s.length() == 0) return 0;
        int count = 0;
        for(int i = (s.length() -1); i >-1 ; i--) {
            if(s.charAt(i) != ' ')  count++;
            if(s.charAt(i) == ' ' && count != 0) return count;
        }
        return count;
        
    }
   public int lengthOfLastWord(String s) {
        
        String[] a = s.split(" ");
        if( a==null || a.length == 0) return 0;
        int count = 0;
        count = a[a.length-1].length();
        return count;
        
    }
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) return false；
        if(root.left ==null && root.right == null && sum ==root.val) return true;
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
        
    }
 public int pathSum(TreeNode root, int sum) {
        if(root == null) return 0;
    	return dfs(root, sum) + pathSum(root.left,sum) + pathSum(root.right,sum);
    }
    	private int dfs(TreeNode root, int sum) {
    		int num = 0;
    		if (root == null) return num;
    		if(root.val == sum) num += 1;
    		num += dfs(root.left, sum-root.val);
    		num += dfs(root.right, sum-root.val);
    		return num;
    	}
public int pathSum(TreeNode root, int sum){
		HashMap<Integer,Integer> preSum = new HashMap<Integer,Integer>();
		preSum.put(0,1);
		return helper(root, 0, sum, preSum);
	}
	private int helper(TreeNode root, int sum, int target, HashMap<Integer,Integer> preSum){
		if (root ==null) {
			return 0;
		}
		sum += root.val;
		int res = 0;
		if (preSum.containsKey(sum-target)) {
		  res = preSum.get(sum-target);
		}
		preSum.put(sum, preSum.getOrDefault(sum,0) +1);
		res += helper(root.left, sum, target,preSum) + helper(root.right, sum,target,preSum);
		preSum.put(sum, preSum.get(sum)-1);// to restall not affect the right side of same layer calculation.
		return res;
	} 

public boolean isAnagram(String s, String t) {
        
        if(s.length() != t.length()) return false;
        if(s.equals(t)) return true;
        int[] freq = new int[26];
        for(int i = 0; i < s.length(); i++) {
            int index = (s.charAt(i) - 'a');
            freq[index] += 1;
        }
        for (int i = 0; i < t.length(); i++) {
            int index2 = (t.charAt(i) - 'a');
            freq[index2] -= 1;
        }
        for(int i = 0; i < 26; i++){
            if(freq[i] != 0) return false;
        }
        return true;
        
    }
public List<Integer> findAnagrams(String s, String p) {
	List<Integer> list = new ArrayList<>();
	if(s == null || p == null || s.length() == 0 || p.length() == 0) return list;
	int[] arr = new int[256];
	for(char c: p.toCharArray()){
		arr[c] += 1;
	}
	int left = 0, right = 0, count = p.length();
	while (right < s.length()) {
		if(arr[s.charAt(right)]) >= 1) {
			count--;
		}
		arr[s.charAt(right)]--;
		right++;
		if(count == 0) list.add(left);
		if ((right - left) == p.length() ) {
			if(arr[s.charAt(left)] >= 0) {
				count++;
			}
			arr[s.charAt(left)]++;
			left++;
		}
	}
	return list;
}
public List<Integer> findAnagrams1(String S, String p) {
	List<Integer> list = new ArrayList<>();
	if (s == null || p == null || s.length() ==0 || p.length() ==0){
		return list;
	}
	int[] res = new int[256];
	int[] res1 = new int[256];
	int ns = s.length(), np = p.length(), i = 0;
	for( char c: p.toCharArray()) {
		res[c] ++;
		res1[c] ++;
	}
	if(m1 == m2) list.add(0);
	for(int i = p.size(); i < s.size(); i++) {
		res[s.charAt(i)] ++;
		res[s.charAt(i) - p.size()]--;
		if(m1 == m2) list.add(i - p.size() +1);
	}
	return list;
	}
}
public int numberOfBoomerangs(int[][] points) {
	int res = 0;
	HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
	for(int i = 0; i < points.length; i++){
		for(int j =0; j< points.length; j++) {
		    if(j==i) continue;
			int d = getDistance(points[i],points[j]);
			map.put(d, map.getOrDefault(d,0) +1);

		}
	
		for(int val: map.values()){
			res += val*(val -1);
		}
		map.clear();
	}
	return res;
}
private int getDistance(int[] a, int[] b) {
	int dx = a[0] - b[0];
	int dy = a[1] - b[1];
	return dx*dx + dy*dy;

    }
    public int majorityElement(int[] nums) {
      HashMap<Integer, Integer> map = new HashMap<>();
      int res = 0;
      for(int i =0; i< nums.length; i++){
          map.put(nums[i], map.getOrDefault(nums[i],0)+1);
          if(map.get(nums[i])> (nums.length/2)) {
              res = nums[i];
              break;
         
      }
    }
    return res;
}

public List<Integer> majorityElement(int[] nums) {
	Int major1 = null, major2 = null, cnt1 = 0, cnt2 = 0;
	for(int num:nums){
		if(num == major1) {
			cnt1 += 1;
		}else if(num == major2){
			cnt2 += 1;
		}
		else if (cnt1 ==0){
			major1 = num;
			cnt1++;
		}
		else if(cnt2 ==0) {
			major2 = num;
			cnt2++;
		}else{
			cnt1--;
			cnt2--;
		}
	}
	cnt1 = cnt2 = 0;
	for(int num: nums){
		if(num==major1) {
			cnt1++;
		}else if(num ==major2){
			cnt2++;
		}

	}
	List<Integer> result = new ArrayList<>();
	if(cnt1 > nums.length /3)  result.add(major1);
	if(cnt2 > nums.length /3)  result.add(major2);
	return result;
}
public int maxProfit(int[] prices) {
	int maxCur = 0; maxSoFar = 0;
	for(int i =1; i<prices.length; i++){
		maxCur = (Math.max(0, maxCur += price[i] - price[i-1]));
		maxSoFar = Math.max(maxCur, maxSoFar);
	}
	return maxSoFar;
}

 public int maxSubArray(int[] nums) {
        int maxcur = nums[0];
        int maxsofar = nums[0];
        for(int i = 1; i< nums.length; i++) {
            maxcur = Math.max(maxcur + nums[i], nums[i]);
            maxsofar = Math.max(maxsofar, maxcur);
        }
        return maxsofar;
        
    }

 public int longestPalindrome(String s) {
        if(s.length() ==0 || s==null) return 0;
        int[] arr = new int[256];
        for(char c: s.toCharArray()) {
            arr[c] += 1;
        }
        int res = 0;
        for(int i =0; i< arr.length; i++){
            if (arr[i] !=0){
                res += (arr[i]/2) * 2;
                
            }
        }
        return res == s.length() ? res:res+1;
        
    }

    public int longestPalindrome(String s) {
       if(s == null || s.length() == 0) return 0;
       HashSet<Character> hs = new HashSet<Character>();
       int count = 0;
       for(int i=0; i<s.length(); i++) {
           if(hs.contains(s.charAt(i))) {
               hs.remove(s.charAt(i));
               count++;
           }else{
               hs.add(s.charAt(i));
           }
       }
       if(hs.isEmpty()) return count*2;
       else return count*2 + 1;
    }
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        if (k <=0) return false;
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(map.containsKey(nums[i])) {
                if ((i - map.get(nums[i])) <= k) return true; 
            }
            map.put(nums[i], i);
        }
        return false;
    }
public String addStrings(String num1, String num2) {
        StringBuilder sb =new StringBuilder();
        int carry = 0;
        for(int i = num1.length()-1, j= num2.length()-1; i>=0|| j>=0|| carry ==1; i--, j--){
            int a = i<0 ? 0: num1.charAt(i) - '0';
            int b = j<0 ? 0: num2.charAt(j) - '0';
            sb.append((a+b+carry)%10);
            carry = (a+b+carry)/10;
        }
    
        return sb.reverse().toString();
    }

    public String toHex(int num) {
        if(num == 0) return "0";
        StringBuilder sb = new StringBuilder();

        int carry = 0;
        while(num!=0){
            int a = (num & 15);
            if (a < 10) sb.append((char)(a + '0'));
            else sb.append((char)(a -10 + 'a'));
            num = (num>>>4);
        }
        return sb.reverse().toString();
    }

public boolean isPalindrome(String s) {
	if(s==null|| s.length() ==0) return true;
	int head =0, tail = s.length() -1;
	char chead, ctail;
	while(head <= tail){
		chead = s.charAt(head);
		ctail = s.charAt(tail);
		if (!Character.isLetterOrDigit(chead)) head++;
		else if( !Character.isLetterOrDigit(ctail)) tail--;
		else {
			if(Character.toLowerCase(chead) != Character.toLowerCase(ctail)) return false;
			head++;
			tail--
		}
		return true;
	}
}
/* This is an iterative version of LCA very impressive! */
public class MyNode{
	TreeNode node;
	MyNode parent;
	boolean visited;
	List<TreeNode> result = new ArrayList<TreeNode>();

	public Mynode (TreeNode node, MyNode parent) {
		this.node = node;
		this.parent = parent;
	}

pubilc TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
	MyNode dummy = new MyNode(null,null);
	MyNode rootNode = new MyNode(root, dummy);
	Stack<MyNode> stack = new Stack<MyNode>();
	stack.push(rootNode);
	while(!stack.isEmpty()) {
		MyNode cur = stack.peek();
		TreeNode node = cur.node;
		MyNode parent = curr.parent;
		if(node ==null || node ==p||node ==q) {
			parent.result.add(node);
			stack.pop();
		}else if(!cur.visited) {
			cur.visited = true;
			stack.push(new MyNode(node.right, cur));
			stack.push(new MyNode(node.left, cur));
		} else if(cur.visited) {
			TreeNode left = cur.result.get(0);
			TreeNode right = cur.result.get(1);
			if(left !=null && right !=null){
				parent.result.add(node);
			}else if(left != null) {
				parent.result.add(left);
			}else{
				parent.result.add(right);
			}
			stack.pop();

		}
	}
	return dummy.result.get(0);
}
public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
	if(k<1 ||t< 0) return false;
	TreeSet<Integer> set = new TreeSet<Integer>();
	for(int i = 0; i < nums.length; i++) {
		int c = nums[i];
		if((set.floor(c) !=null && c<=t + set.floor(c)) || (set.ceiling(c)!=null && set.ceiling(c) -t <=c )) {
			return true;
		}
		set.add(c);
		if(i >= k) set.remove(nums[i-k]);
	}
	return false;
}
public static void print(Map<String,Integer> wordMap){
	List<Map.Entry<String,Integer>> list = new ArrayList<>();
	list.addAll(wordMap.entrySet());
	Collections.sort(list,new Comparator<Map.Entry<String,Integer>>(){
	  public int compare(Map.Entry<String,Integer> o1,Map.Entry<String,Integer> o2){
				return o2.getValue().compareTo(o1.getValue());
			}
		}
}

public int rob(int[] nums){
	int preno = 0;
	int preyes = 0;
	for(int n: nums){
		int curno = Math.max(preno,preyes);
		int curyes = preno + n;
		preno = curno;
		preyes = curyes;
	}
	return Math.max(curno,curyes);
}
public int rob(int[] nums) {
	int pre = 0;
	int cur = 0;
	for(int i = 0; i<nums.length; i++) {
		int temp = Math.max(pre + nums[i], cur);
		pre = cur;
		cur = temp;
	}
	return cur;
}

 public String reverseVowels(String s) {
        if(s == null || s.length()==0) return s;
	String vowels = "aeiouAEIOU";
	int first = 0, end = s.length() -1;
	char[] array = s.toCharArray();
	while(first < end){
		while(first < end && vowels.indexOf(array[first]) == -1){
			first++;
		}
		while(first <end && vowels.indexOf(array[end]) == -1) {
			end--;
		}
		char temp = array[first];
		array[first] = array[end];
		array[end] = temp;
		first++;
		end--;
	}
	return new String(array);
}
public List<String> binaryTreePaths(TreeNode root) {
	List<String> res = new ArrayList<>();
	StringBuilder sb = new StringBuilder();
	helper(res, root, sb);
	return res;
}
private void helper(List<String> res, TreeNode root, StringBuilder sb) {
	if(root ==null) return;
	int len = sb.length();
	sb.append(root.val);
	if(root.left == null && root.right == null) {
		res.add(sb.toString());
	}else {
		sb.append("->");
		helper(res, root.left, sb);
		helper(res, root.right, sb);
	}
	sb.setLength(len);
}
public List<String> binaryTreePaths(TreeNode root) {
	ArrayList<String> finalresult = new ArrayList<String>();
	if(root == null) return finalresult;
	ArrayList<String> curr = new ArrayList<String>();
	ArrayList<ArrayList<String>> results = new ArrayList<ArrayList<String>>();
	dfs(root,results, cur);
	for(ArrayList<String> a : results) {
		StringBuilder sb = new StringBuilder();
		sb.append(a.get(0));
		for(int i = 1; i< a.size(), i++){
			sb.append("->" + a.get(i));
		}
		finalresult.add(sb.toString());
	}
	return finalresult;
}
private void dfs(TreeNode root, ArrayList<ArrayList<String>> res, ArrayList<String> cur) {
	cur.add(String.valueOf(root.val))；
	if(root.left ==null && root.right == null) {
		res.add(cur);
		return;
	}
	if(root.left != null) {
		ArrayList<String> tmp = new ArrayList<String>(cur);
		dfs(root.left, res,tmp);
	}
	if(root.right != null) {
		ArrayList<String> tmp = new ArrayList<String>(cur);
		dfs(root.right, res,tmp);
	}
}

 public int findRadius(int[] houses, int[] heaters) {
        int left = 0;
        int right = heaters.length -1;
        Arrays.sort(heaters);
        int result = Integer.MIN_VALUE;
        for(int house : houses) {
            int index = binarysearch(heaters, house);
            int dist1 = index >= 1 ? house - heaters[index -1] : Integer.MAX_VALUE;
            int dist2 = index < heaters.length? heaters[index] - house : Integer.MAX_VALUE;
            result = Math.max(result, Math.min(dist1, dist2));
            
            
        }
       return result;
        }
       private int binarysearch(int[] array, int target){
           int left = 0;
           int right = array.length -1;
           while(left <= right){
               int mid = left + (right -left) /2;
               if(target == array[mid]) return mid;
               else if( target < array[mid]) right = mid -1;
               else left = mid + 1;
           }
           return left;
    ｝
    //MorrisTraversal O(n) time and O(1) space
 
    class BinaryTree{
    	tNode root;
    	void MorrisTraversal(tNode root){
    		tNode current, pre;
    		if(root == null) return;
    		current = root;
    		while(current != null){
    			if(current.left == null){
    				System.out.print(current.data + " ");
    				current = current.right;
    			}
    			else{
    				pre = current.left;
    				while(pre.right != null && pre.right != current)
    					pre = pre.right;
    				if(pre.right == null){
    					pre.right = current;
    					current = current.left;
    				}
    				else{
    					pre.right = null;
    					System.out.print(current.data + " ");
    					current = current.right;
    				}
    			}
    		}
    	}
    }

public int[] findMode(TreeNode root){
	inorder(root);
	models = new int[modeCount];
	modeCount = 0;
	currentCount = 0;
	inorder(root);
	return modes;
}
private int currVal;
private int currCount = 0;
private int maxcount = 0;
private int modeCount = 0;
private int[] modes;

private void handlevalue( int val) {
	if(val != currval) {
		curr = val;
		currCount = 0;
	}
	currCount++;
	if(currentCount > maxcount){
		maxcount = currCount;
		modeCount = 1;
	}else if(currentCount == maxcount){
		if(models != null) model[modecount] = currval;
		modecount++;
	}

}
private void inorder(TreeNode root){
	if(root == null) return;
	inorder(root.left);
	handlevalue(root.val);
	inorder(root.right);
}
public class Solution {
    Map<Integer, Integer> map;
    int max = 0;
    public int[] findMode(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<>();
        if(root == null) return new int[0];
        this.map = new HashMap<>();
        inorder(root);
        for(int key:map.keySet()) {
            if(max == map.get(key)) res.add(key);
        }
       
        int[] result = new int[res.size()];
        for(int i = 0; i< res.size(); i++){
            result[i] = res.get(i);
        }
        return result;
        
    }
    private void inorder(TreeNode node){
        if(node == null) return;
        inorder(node.left);
        map.put(node.val, map.getOrDefault(node.val,0) + 1);
        max = Math.max(max, map.get(node.val));
        inorder(node.right);
    }

public List<Integer> postorderTraversal(TreeNode root) {
	List<Integer> res = new ArrayList<>();
	if(root == null) return res;
	Stack<TreeNode> stack = new Stack<>();
	TreeNode p = root;
	while(!stack.isEmpty() || p != null){
		if(p != null){
			stack.push(p);
			//res.add(n.val);
			p = p.left;
		}else{
		TreeNode n = stack.pop();
		res.add(n.val);//
		p = n.right;
	}
}
return res;
}

public List<Integer> preorderTraversal(TreeNode root){
	List<Integer> res = new ArrayList<>();
	if(root == null) return res;
	Stack<TreeNode> stack = new Stack<>();
	stack.push(root);
	while(!stack.isEmpty()){
		TreeNode n = stack.pop();
		res.add(n.val);
		if(n.right != null){
			stack.push(n.right);
			}
		if(n.left != null){
			stack.push(n.left);
		}
	}
	return res;
	}

public List<Integer> postorderTraversal(TreeNode root){
	List<Integer> res = new ArrayList<>();
	if(root == null) return res;
	Stack<TreeNode> stack = new Stack<>();
	TreeNode p = root;
	while(!stack.isEmpty() || p != null){
		if(p != null){
			stack.push(p);
			p = p.left;
		}else{
			TreeNode n = stack.peek();	
			if(n.right != null && pre!= n.right){
				p = p.right;
			}else{
				res.add(n);
				stack.pop();
				pre = n;
			}
			}
	}
return res;
}  
