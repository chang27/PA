package edu.uci.ics.textdb.exp.dictionarymatcher;

import java.util.*;

/**
 * Created by Chang on 9/19/17.
 */
public class ArrayHopper {
    public final String FAILURE = "failure";
    public final String OUT = "out";

    public ArrayList<Integer> findHops (int[] nums){
        ArrayList<Integer> result = new ArrayList<>();
        if(nums == null || nums.length == 0) return result;
        int curMax = 0;
        int lastMax = 0;
        int index = 0;
        int step = 0;
        int i;
        for(i = 0; i < nums.length; i++) {
            if(i > curMax) break;
            if(i > lastMax){
                step++;
                lastMax = curMax;
                result.add(index);
            }
            int sum = i + nums[i];
            if(sum > curMax){
                curMax = sum;
                index = i; // this is greedy to find the largest step it can take in the current range of reaching area
            }

        }
        if(i > lastMax && ! result.isEmpty() && result.get(result.size() - 1) != index){
            result.add(index);
        }
        if(lastMax >= nums.length - 1) return result;

        return null;
    }

    public static void main(String[] args) {
        int[] in = {2,4, 3, 1,1,1,0};
        ArrayHopper ah = new ArrayHopper();
        System.out.println(ah.findHops(in).toString());
    }
}
