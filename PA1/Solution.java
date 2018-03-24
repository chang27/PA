import java.io.*;
import java.util.*;



public class Solution {
    public static String getPermutation(int n, int k) {
	int[] nums = new int[n];
	for(int i = 1; i<=n; i++){
		nums[i-1] = i;
	}
	List<List<Integer>> res1 = permute(nums);
	//Collections.sort(res1);
	List<Integer> r = res1.get(k-1);
	return r.toString();
	
	
		
        
    }
	
    public static List<List<Integer>> permute(int[] nums) {
	List<List<Integer>> res = new ArrayList<List<Integer>>();
	List<Integer> list = new ArrayList<Integer>();
	helper(res, list, nums);
	return res;
	}
    private static void helper(List<List<Integer>> res, List<Integer> list, int[] nums){
	if(list.size() == nums.length){
		res.add(new ArrayList(list));
		return;
		}
	for(int i=0; i < nums.length; i++){
		if(list.contains(nums[i])) continue;
		list.add(nums[i]);
		helper(res, list, nums);
		list.remove(list.size() -1);
	}
}
     public static void main(String[] args){
		int n = 3;
		int k = 4;
		String a = getPermutation(n, k);
		System.out.println(a.charAt(0));
}
}



