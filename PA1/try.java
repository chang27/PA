import java.util.*;
import java.io.*;
public class solution {
	public static List<String> subset(String s){
		char[] c = s.toCharArray();
		int totalnum = 1<<c.length;
		List<String> res = new ArrayList<String>();
		int n = c.length;
		for(int i = 1; i< totalnum; i++){
			StringBuilder sb = new StringBuilder();
			for(int j = 0; j<n; j++){
				if((i & (1<<j)) != 0){
					sb.append(c[i]);
				}
			}
			res.add(sb.toString());
		}
		return res;
	}
	public static void main(String[] args){
		String input = "abc";
		int result = subset(input);
		Systems.out.println(result);
	}
}