import java.util.*;
import java.io.*;
public class st {
	public static String getst (String input){
	String out = input.replaceAll("[^A-Z0-9a-z]+","");
	return out;
	}
	public static boolean ifcontain(String a, String b){
		String c = getst(a);
		if(c.length() < b.length()) return false;
		int j = 0;	
		for(int i = 0; i < c.length(); i++){
			if(c.charAt(i) == b.charAt(j)) j++;
			if(j == b.length()) return true;
		}
		return false;
		}	
		
public static void main(String[] args){
	String in = "(abcde)(hell0";
	String b = "helll";
	boolean j = ifcontain(in, b);
	System.out.println(j);
}
}
