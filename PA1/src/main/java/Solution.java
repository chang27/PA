import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by Chang on 11/4/17.
 */
public class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int size = k;

        int low = 0;
        int high = arr.length - 1;
        while(low < high){
            int mid = low + (high - low)/2;
            if(arr[mid] <= x){
                low = mid + 1;
            }else high = mid;
        }
        //high is the first element larger or equal to x

        if(low > 0){
            low--;
        }
        //low is the first element smaller than x
        System.out.println(low);

        while(low > 0 && high < arr.length && k > 1){
            if(x - arr[low] <= arr[high] - x){
                low--;
            }else{
                high++;
            }
            k--;
        }

        System.out.println("value of k " + k);
        System.out.println("value of low " + low);

        while(k > 1 && high == arr.length){
            low--;
            k--;
        }

        System.out.println("value of k " + k);
        System.out.println("value of low " + low);

        if(low >= 0 && low + size < arr.length && (x - arr[low]) > (arr[low + size] - x)){
            low++;
        }
//        System.out.println(low);
//        System.out.println(k);

        List<Integer> res = new ArrayList<>();
        for(int i = low; i < low + size; i++){
            res.add(arr[i]);
        }
        return res;


    }
    static class ListNode{
        int val;
        ListNode next;
        public ListNode(int val){
            this.val = val;
            this.next = null;
        }
    }

    public static ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode cur = head;
        ListNode newHead = null;
        ListNode p = null;
        while(cur != null && cur.next != null) {
            System.out.println("77 " + cur.val);
            if(cur.val == cur.next.val){
                while(cur != null && cur.next != null && cur.val == cur.next.val){
                    cur = cur.next;
                    System.out.println("82 " + cur.val);
                }
                cur = cur.next;
                if(newHead == null){
                    newHead = cur;
                    p = newHead;
                }else{
                    p.next = cur;
                    // cur = cur.next;
                }
            }else{
                cur = cur.next;
                if(newHead == null){
                    newHead = cur;
                    p = newHead;
                }else{
                    p.next = cur;
                    p = p.next;
                }
                System.out.println("100 " + cur.val);

            }
        }
        return newHead;
    }

    public static int minOne(int[] input, int target){
        int i = 0, j = 0;
        int n = input.length;
        int cnt = 0;
        while(i < n){
            if(input[i] == target){
                cnt += i - j;
                j++;
              //  int tmp = input[i];

            }
            i++;
        }
        return cnt;
    }

    public static int minResult(int[] input){
        int result1 = minOne(input, 1);
        int result0 = minOne(input, 0);
        return Math.min(result1, result0);
    }



    public static void main(String[] args){
//        Solution s = new Solution();
//        int[] input = {1,2,3,4,5};
//        List<Integer> res = s.findClosestElements(input, 4, 3);
//        System.out.println(res.toString());
//        ListNode n1 = new ListNode(1);
//        n1.next = new ListNode(1);
//        n1.next.next = new ListNode(2);
//        n1.next.next.next = new ListNode(3);
//        System.out.println(n1.next.next.val);
 //       System.out.println(n1.next.next.next.val);
     //   ListNode result = deleteDuplicates(n1);
        int[] in = {0, 1, 1, 0, 0, 1};
        int res = minResult(in);
        System.out.println(res);

    }
}
