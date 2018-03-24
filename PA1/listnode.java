public class listnode {
	public int val;
	public listnode next;
	public listnode (int val) {
		this.val = val;
		this.next = null;
	}
}
public class reverse(listnode head){
	public listnode prev = null;
	while(head !=null){
		listnode tmpnext = head.next;
		head.next = prev;
		prev = head;
		head = tmpnext;
	}
	return prev;
}
public ListNode swapPairs(ListNode head){
	if(head ==null ||head.next ==null) return head;
	ListNode newhead = head.next;
	head.next = swapPairs(head.next.next);
	newhead.next = head;
	return head;
}
public ListNode swapPairs(ListNode head){
	if(head==null || head.next ==null) return head;
	ListNode dummy = new ListNode(0);
	dummy.next = head;
	ListNode current = dummy;
	while(current.next !=null || current.next.next !=null) {
		ListNode newhead = current.next.next;
		newhead.next = current.next;
		current.next = newhead;
		current = current.next.next;
	}
	return dummy.next;
}
public class Dlistnode {
	putblic int val;
	public Dlistnode next, prev;
	Dlistnode (int val) {
		this.val = val;
		this.prev = this.next = null;
	} 
}

public class reverseD(Dlistnode) {
	putlic Dlistnode curr = null;
	while (head != null) {
		curr = head.next;
		head.next = head.prev;
		head.prev = curr; 
		head = curr;
	}

}

public class TreeNode {
	public int val;
	public TreeNode left, right;
	public TreeNode(int val) {
		this.val = val;
		this.left = null;
		this.right = null;
	}
}

public int atoi(String str) {
	if (str == null || str.length() < 1)
		return 0;
	str = str.trim();
	char flag = '+';
	int i = 0;
	if (str.charAt(0) == '-'){
		flag = '-';
		i++;
	}else if (str.charAt(0) =='+'){
		i++;
	}
	int result = 0;
	while(str.length() > i && str.charAt(i) >= '0' && str.length()<='9'){
		if(Integer.MAX_VALUE /10 < result || Integer.MAX_VALUE/10 == result && Integer.MAX_VALUE%10 < (str.charAt(i) - '0'))
			return flag == '-' ? Integer.MIN_VALUE : Integer.MAX_VALUE;
		result = result * 10 + (str.charAt(i) - '0');
		i++;
	}
	if(flag == '-')
		result = -result;
	return result;
}
public int[] twoSum(int[] numbers, int target) {
	HashMap<Integer, Integer> map = new HashMap<Integer,Integer>();
	int[] result = new int[2];
	for (int i = 0; i < numbers.length; i++) {
		if (map.containsKey(number[i])) {
			int index = map.get(numbers[i]);
			result[0] = index;
			result[1] = i;
			break;
		}else {
			map.put(target - numbers[i], i);
		}
		}
	return result;
	}
}public class Solution {
    public int[] twoSum(int[] numbers, int target) {
        if (numbers == null || numbers.length == 0) 
            return null;
        int i = 0;
        int j = numbers.length - 1;
        while(i<j){
            int x = numbers[i] + numbers[j];
            if (x < target) {
                i = i+1;
            }else if (x > target) {
                j = j-1;
            }else{
                int[] result = new int[2];
                result[0] = i+1;
                result[1] = j+1;
                return result;
            }
        
            
        }
        return null;
        
    }
}
public String addBinary(String a,String b) {
	if(a == null || a.length() ==0)
		return b;
	if(b == null || b.length()==0)
		return a;
	int i = a.length() - 1;
	int j = b.length() - 1;
	int carry = 0;
	StringBuilder res = new StringBuilder();
	while(i>=0 && j>=0){
		int digit = (int)((a.charAt(i) - '0')+(b.charAt(j) - '0') + carry);
		carry = digit / 2;
		digit = digit % 2;
		res.append(digit);
		i= i-1;
		j=j-1;

	}
	while(i >= 0){
		int digit = (int)((a.charAt(i) - '0')+ carry);
		carry = digit/2;
		digit = digit%2;
		res.append(digit);
		i = i - 1;
	}
	while(j >= 0){
		int digit = (int)((b.charAt(i) - '0')+ carry);
		carry = digit/2;
		digit = digit%2;
		res.append(digit);
		j = j - 1;
	}
	if (carry >0){
		res.append(carry);
	}
	return res.reverse().toString();
}
public int maxrotaterunction(int[] A) {
	if(A == null || A.length < 1) {
		return  0;
	}
	int n = A.length;
	int sum = 0;
	int pre = 0;
	for (int i = 0; i < n; i++) {
		sum += A[i];
		pre += i * A[i];
	}
	int max = pre;
	for (int i = 1; i < n; i++) {
		int pre = pre + sum - n * A[n-i];
		max = Math.max(max, pre);
	}
}public int reverse(int x) {
        long rev = 0;
        while (x != 0 ) {
            int tail = x % 10;
            rev = rev * 10 + tail;
            x = x / 10;
            if(rev > Integer.MAX_VALUE || rev < Integer.MIN_VALUE)
                return 0;
        }
        return (int)rev;
    }
    public int reversebits(int n) {
    	int res = 0;
    	for (int i = 0; i < 32; i++){
    		if (n & 1 == 1){
    			res = (res<<1) + 1;
    		}else {
    			res = res << 1;
    		}
    		n = n >> 1;
    	}
    	return res;
    }
public int reverseBits(int n) {
    int res = 0;
    for (int i = 0; i < 32; i++){
        res = (res << 1) + (n & 1);
    	n = n >> 1;
    }
    return res;
 }
 /*# java.lang.Character.getNumericValue(char ch)#*/
 public int findNthDigit(int n) {
        int start = 1;
        int len = 1;
        long count = 9;
        long m = n;
        while (m > len*count) {
            m = m - len*count;
            start = start*10;
            len = len + 1;
            count = count * 10;
            n = (int) m;
        }
        start = start + (n-1)/len;
        String s = Integer.toString(start);
        int result = Character.getNumericValue(s.charAt((n-1)%len));
        return result;
        
    }
    public ListNode removeElements( ListNode head, int val) {
    	if (head == null) return null;
    	ListNode next = removeElements(head.next, val);
    	if(head.val == val) return next;
    	head.next = next;
    	return head;
    }
 public class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode helper = new ListNode(0);
        helper.next = head;
        ListNode p = helper;
        while (p.next != null) {
            if (p.next.val == val){
                p.next = p.next.next;
            }else{
                p=p.next;
            }
            
        }
      return helper.next;  
    }
}
/*the next node will not have reference, then the Java System will delete the node without reference automatically.*/
public void deleteNode(ListNode node){
	node.val = node.next.val;
	node.next = node.next.next
}
public class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return null;
        ListNode helper = new ListNode(0);
        helper.next = head;
        ListNode p = helper.next;
        while(p.next != null){
            if(p.val !=p.next.val){
                p = p.next;
            }
            else {
                p.next = p.next.next;
            }
        }
        return helper.next;
    }
}
public ListNode removeNthFromEnd(ListNode head, int n) {
	if (head == null) return null;
	ListNode fast = head;
	ListNode slow = head;
	for (int i = 0; i<n; i++){
		fast=fast.next;
	}
	if(fast == null) {
		head = head.next;
		return head;
	}
	while (fast.next !=null){
		fast = fast.next;
		slow = slow.next;
	}
	slow.next = slow.next.next;
	return head;
}

public boolean isPalindrome(ListNode head) {
	ListNode fast = head, slow = head;
	while (fast !=null && fast.next !=null) {
		fast = fast.next.next;
		slow = slow.next;
	}
	if (fast != null){
		slow = slow.next;
	}
	slow = reverse(slow);
	fast = head;
	while(slow != null) {
		if (fast.val != slow.val) {
			return false;
		}
		fast = fast.next;
		slow = slow.next;
	}
	return true;
}
public ListNode reverse(ListNode head) {
	ListNode prev = null;
	while(head != null) {
		ListNode next = head.next;
		head.next = prev;
		prev = head;
		head = next;
	}
	return prev;
}
/*bitwise & no higher priority than +   */
 public int hammingDistance(int x, int y) {
    int xor = xˆy;
    int count = 0;
    for (int i = 0; i< 32; i++) {
        count = count + ((xor >> i) & 1);
    }
    return count;
}
/*string to char to character array */
public Character[] toCharacterArray( String s ) {

   if ( s == null ) {
     return null;
   }

   int len = s.length();
   Character[] array = new Character[len];
   for (int i = 0; i < len ; i++) {
      array[i] = new Character(s.charAt(i));
   }

   return array;
}
public String reverseString(String s) {
	if(s == null)||s.length() == 0) return "";
	char[] array = s.toCharacterArray();
	int start = 0;
	int end = s.length() - 1;
	while(start <= end){
		char tmp = array[start];
		array[start] = array[end];
		array[end] = tmp;
		start++;
		end--
	}
	String result =.valueOf(array);
	result result;
}
/*count the head of the boat */
public int countBattleships(char[][] board) {
	int m = board.length;
	int (m ==0) return 0;
	int n = board[0].length;
	int count = 0;
	for (int i=0; i<m; i++){
		for(int j = 0; j< n; j++){
			if(board[i][j] = '.') continue;
			if(i > 0 && board[i-1][j] == 'X') continue;
			if(j > 0 && board[i][j-1] == 'X') continue;
			count = count + 1;

		}
	}
	return count;
}
public int islandPerimeter(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        int inum = 0; 
        int nei = 0;
        if(n==0) return 0;
        for (int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                if (grid[i][j] ==0) continue;
                if (grid[i][j] == 1) {
                    inum = inum + 1;
                    if(i < (n-1) && grid[i+1][j] ==1) nei = nei + 1;
                    if(j < (m-1) && grid[i][j+1] ==1) nei = nei + 1;
                }
            }
        }
        int result = inum *4 - nei*2;
        return result;
        
}
public int islandPerimeter(int[][] grid) {
	int n = grid.length;
	int m = grid[0].length;
	int res = 0;
	if( n == 0 ) return 0;
	for (int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++){
			if (grid[i][j] == 0) continue;
			res = res + 4;
			if (i >0 && grid[i-1][j] == 1) res = res - 2;
			if (j >0 && grid[i][j-1] == 1) res = res - 2;
		}
	}
	return res;

}
public List<Integer> findDisappearedNumbers( int[] nums) {
	List<Integer> ret = new ArrayList<Integer>();

	for(int i = 0; i < nums.length; i++) {
		int val = Math.abs(nums[i]) - 1;
		if(nums[val] > 0) {
			nums[val] = -nums[val];
		}
	}

	for(int i=0; i<nums.length; i++) {
		if(nums[i] > 0) {
			ret.add(i+1);
		}
	}
	return ret;
}
publist List<Integer> findDisappearedNumbers2(int[] nums) {
	List<Integer> res = new ArrayList<>();
	int n = nums.length;
	for (int i = 0; i<nums.length; i++) nums[(nums[i]-1)%n] += n；
	for(int i = 0; i<nums.length; i++) if (nums[i] <=n) res.add(i+1);
	return res;
}
public char findTheDifference(String s, String t) {
	char c = 0;
	for (int i=0; i<s.length(); i++) {
		c ˆ=s.charAt(i);
	}
	for (int i = 0; i < t.length(); i++) {
		c ˆ= t.charAt(i);
	}
	return c;
}
public char findTheDifference2(String s, String t) {
	int ret = 0;
	for(int i=0; i<s.length(); i++) {
		ret -= (int)s.charAt(i);
	}
	for(int i=0; i<t.length(); i++) {
		ret += (int)t.charAt(i);
	}
	return (char)ret;
}
int getSum(int a, int b) {
	if (a==0) return b;
	while(b!=0){
		carry = (a & b) << 1;
		sum = a ^ b;
		return getSum(sum, carry);
	}
}
 public ListNode swapPairs(ListNode head){
	if(head==null || head.next ==null) return head;
	ListNode dummy = new ListNode(0);
	dummy.next = head;
	ListNode current = dummy;
	while(current.next !=null || current.next.next !=null) {
		ListNode newhead = current.next.next;
		ListNode reverse = current.next;
		reverse.next = newhead.next;
		current.next = newhead;
		current.next.next = reverse;
		current = current.next.next;
	}
	return dummy.next;
}

public boolean isSymmetric(TreeNode root) {
	if(root ==null) return true;
	return ishelp(root.left, root.right);
}
 private boolean ishelp (TreeNode left, TreeNode right) {
 	if(left==null && right ==null) return true;
 	if(left ==null || right ==null) return false;
 	if(left.val != right.val) return false;
 	return ishelp(left.left, right.right) && ishelp(left.right, right.left);
 	}
 }
public static List<List<Integer>> generate(int numRows){
	List<List<Integer>> res = new ArrayList<List<Integer>>();
	if (numRows ==0) return res;
	for(int i = 0; i< numRows; i++){
		ArrayList<Integer> row = new ArrayList<Integer>();
		row.add(1);
		for(int j=1; j<i; j++){
			ArrayList<Integer> prerow = res.get(i-1);
			int tmp = prerow.get(j-1) + prerow.get(j);
			row.add(tmp);
		}
		if (i != 0) row.add(1);
		res.add(row);
	}
	return res;
}
 public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<Integer>();
        if(rowIndex < 0) return res;
        res.add(1); 
        for(int i =1; i<= rowIndex; i++){
            for(int j = i-1; j>=1; j--){
                res.set(j, res.get(j-1) +res.get(j));
            }
         res.add(1);  
        }
        return res;
        
    }
public boolean isBalanced(TreeNode root){
	if (root ==null) return true;
	int left = depth(root.left);
	int right = depth(root.right);
	if (Math.abs(left-right) >1 || !isBalanced(root.left) || isBalanced(root.right)) return false;
	else return true;

}
public int depth(TreeNode root){
	if (root ==null) return 0;
	else return Math.max(depth(root.left), depth(root.right)) + 1;
}
public boolean isBalanced(TreeNode root){
	if(root ==null) return true;
	return dfsHeight(root != -1);
}
public int dfsHeight(TreeNode root) {
	if(root ==null) return 0;
	int leftHeight = dfsHeight(root.left);
	if (leftHeight == -1) return -1;
	int rightHeight = dfsHeight(root.right);
	if(rightHeight = -1) return -1;
	if(abs(leftHeight - rightHeight) > 1) return -1;
	return Math.max(leftHeight, rightHeight) + 1;

}

public int arrangeCoins(int n){
	int start = 0;
	int end = n;
	int mid = 0;
	while(start <= end){
		mid = start + (start + end)/2;
		if(0.5*mid*mid + 0.5*mid) <= n) {
			start = mid + 1;
		}else {
			end = mid -1;
		}
	}
		return start -1;
}
 public boolean isValidSudoku(char[][] board) {
        if (board == null || board.length != 9 || board[0].length != 9)
		        return false;
        HashSet<Character> set = new HashSet<Character>();
        for(int i =0; i<9; i++){
            for(int j = 0; j<9; j++){
                if (board[i][j] == '.') continue;
                if(set.contains(board[i][j])) return false;
                set.add(board[i][j]);
            }
            set.clear();
        }
        for(int j = 0; j<9; j++){
            for(int i = 0; i<9; i++){
                if(board[i][j] =='.')  continue;
                if(set.contains(board[i][j])) return false;
                set.add(board[i][j]);
            }
            set.clear();
        }
        for(int k = 0; k < 9; k++){
            for(int i=(k / 3)*3; i< ((k / 3)*3+3); i++) {
                for(int j = (k%3)*3; j< (k%3)*3+3; j++){
                    if(board[i][j] == '.') continue;
                    if(set.contains(board[i][j])) return false;
                    set.add(board[i][j]);
                }
            }
        set.clear();
        }
        return true;
    }
public String countAndSay(int n) {
        if(n<1) return "";
        String res = "1";
       
        for(int i = 1; i < n; i++){
            int count = 1;
            StringBuilder sb = new StringBuilder();
            for(int j = 1; j <res.length(); j++){
                if (res.charAt(j) == res.charAt(j-1)){
                    count++;
                }else{
                    sb.append(count).append(res.charAt(j-1));
                    count = 1;
                }
            }
            sb.append(count).append(res.charAt(res.length()-1));
            res = sb.toString();
        }
        return res;
        
    }
public String countAndSay(int n) {
	StringBuilder cur = new StringBuilder("1");
	StringBuilder pre;
	
	for(int i = 1; i<n; i++){
		int count = 1;
		pre = cur;
		cur = new StringBuilder();
		for(int j = 1; j< pre.length(); j++){
			if(pre.charAt(j) ! = pre.charAt(j-1)){
				cur.append(count).append(pre.charAt(j-1));
				count = 1;
			}
			else count++;
		}
		cur.append(count).append(pre.charAt(pre.length() -1));
	}
	return curr.toString();

}
  public int searchInsert(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;
        while(start <= end) {
            int mid = start + (end - start)/2;
            if(nums[mid] < target) start = mid + 1;
            else if(nums[mid] == target) return mid;
            else end = mid-1;
            
        }
        return start;
        
    }
    public int mySqrt(int x) {
        int low = 1;
        int high = x;
        while(low <= high) {
            int mid = low + (high - low)/2;
            if ((long) mid*mid == x) return mid;
            else if ((long) mid*mid < x) low = mid+1;
            else if ((long) mid*mid > x) high = mid-1;
        }
        return high;
    }
    Stack<Character> stack = new Stack<Character>();
        for (char c : s.toCharArray()) {
            if(c=='(') 
                stack.push(')');
            
            if(c=='{') 
                            stack.push('}');
            }
            if(c == '[') {
                stack.push(']');
            }
            else if (stack.isEmpty() || stack.pop() != c)
                return false;
            }
            return stack.isEmpty();
    }

    public String longestCommonPrefix(String[] strs) {
    	if(strs == null || strs.length ==0) {
    		return "";
    	}
    	String prefix = strs[0];
    	for(int i = 1; i < strs.length; i++){
    		int j = 0;
    		while(j < strs[i].length() && j < prefix.length() && strs[i].charAt(j) == prefix.charAt(j)){
    			j++;
    		}
    		prefix = prefix.substring(0,j);
    	}
    	return prefix;
    }
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null) return null;
        ListNode a = headA;
        ListNode b = headB;
        while(a != b){
            a = a.next;
            b = b.next;
            if (a == null && b != null) a = headB;
            if (b == null && a != null) b = headA;
        }
       return a; 
    }
}
public void rotate(int[] nums, int k) {
        if( nums.length ==0 || nums == null || k<=0) return;
        int[] copy = nums.clone();
        int n = nums.length;
        for (int i = 0; i< nums.length; i++){
            nums[(i+k)%n] = copy[i];
        }
        
    }
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