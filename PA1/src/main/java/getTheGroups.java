import java.util.Arrays;

/**
 * Created by Chang on 11/8/17.
 */
public class getTheGroups {

    public static String friends = "Friends";
    public static String union = "Unions";
    public int[] parent;
    public int[] size;
    public void initialize(int n){
        parent = new int[n];
        size = new int[n];
        for(int i = 0; i < n; i++){
            parent[i] = i;
            size[i] = 1;
        }
      //  Arrays.fill(size, 1);
    }

    public int find(int i){
        while(parent[i] != i){
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }
    boolean isValid(int i){
        if(i >= 0 && i < parent.length){
            return true;
        }else{
            throw new IllegalArgumentException("this index is not valid" + i);
        }
    }
    public void union(int i, int j){
        if(!isValid(i) || !isValid(j)){
            return;
        }
        int rooti = find(i);
        int rootj = find(j);
        if(rooti == rootj)  return;
        if(size[rooti] >= size[rootj]){
            parent[rootj] = rooti;
            size[rooti] += size[rootj];
        }else{
            parent[rooti] = rootj;
            size[rootj] += size[rooti];
        }
    }
   public int getSize(int i, int j){
        if(!isValid(i) || !isValid(j)){
            throw new IllegalArgumentException("this index is not valid" + i);

        }
        int root1 = find(i);
        int root2 = find(j);
        if(root1 == root2){
            return size[root1];
        }
        return size[root1] + size[root2];
   }
   public static void main(String[] args){
       getTheGroups gtg = new getTheGroups();
       gtg.initialize(5);
       gtg.union(2, 3);
       gtg.union(3, 4);
       gtg.union(4, 1);
       int re = gtg.getSize(2, 0);
       System.out.println(re);


   }
}
