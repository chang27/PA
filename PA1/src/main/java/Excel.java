import java.util.HashMap;
import java.util.Map;

/**
 * Created by Chang on 3/2/18.
 */
public class Excel {
    Cell[][] table;

    public Excel(int H, char W){
        table = new Cell[H+1][W - 'A' + 1];
    }

    public void set(int r, char c, int v){
        if(table[r][c - 'A'] == null) table[r][c - 'A'] = new Cell(v);
        else table[r][c - 'A'].setVal(v);
    }

    public int get(int r, char c){
        if(table[r][c - 'A'] == null) return 0;
        else return table[r][c - 'A'].getValue();
    }

    public int sum(int r, char c, String[] str){
        if(table[r][c - 'A'] == null){
            table[r][c - 'A'] = new Cell(str);
        }else{
            table[r][c - 'A'].setFormular(str);
        }
        return table[r][c - 'A'].getValue();
    }
    private class Cell{
        int val = 0;
        Map<Cell, Integer> formular = new HashMap<>();
        public Cell(int val){
            formular.clear();
            this.val = val;
        }
        public Cell(String[] str){
            setFormular(str);
        }

        public void setVal(int val){
            formular.clear();
            this.val = val;
        }

        public void setFormular(String[] str){
            formular.clear();
            for(String s : str){
                if(s.indexOf(":") < 0){
                    int[] pos = getPos(s);
                    addFormularCell(pos[0], pos[1]);

                }else{
                    String[] pos = s.split(":");
                    int[] start = getPos(pos[0]);
                    int[] end = getPos(pos[1]);
                    for(int r = start[0]; r <= end[0]; r++){
                        for(int c = start[1]; c <= end[1]; c++){
                            addFormularCell(r, c);
                        }
                    }

                }
            }
        }


        private int[] getPos(String s){
            int[] pos = new int[2];
            pos[1] = s.charAt(0) - 'A';
            pos[0] = Integer.parseInt(s.substring(1));
            return pos;
        }

        private void addFormularCell(int r, int c){
            if(table[r][c] == null) table[r][c] = new Cell(0);
            Cell rangeCell = table[r][c];
            formular.put(rangeCell, formular.getOrDefault(rangeCell, 0) + 1);
        }

        private int getValue(){
            if(this.formular.isEmpty()) return this.val;
            int sum = 0;
            for(Cell c : formular.keySet()){
                sum += c.getValue()*formular.get(c);
            }
            return sum;
        }
    }
}
