public class perfectsqure{
 public static int numSquares(int n) {
        int[] dp = new int[n+1];
        dp[0] = 0;
	for(int i = 1; i < n+1; i++){
		dp[i] = i;
		}
        for(int i = 1; i <= n; i++){
            for(int j = 1; j*j <=i; j++){
                dp[i] = Math.min(dp[i], dp[i-j*j] + 1);
		System.out.println(i + ":" + dp[i]);
            }
        }
        return dp[n];
    
	}
public static void main(String[] args){
	int n = 6;
	int out = numSquares(n);
	System.out.println(out);
	}
}
