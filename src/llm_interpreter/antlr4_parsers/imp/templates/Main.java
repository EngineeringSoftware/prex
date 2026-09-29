import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import java.util.Map;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;

public class Main {
    public static void main(String[] args) throws Exception {
        CharStream input = CharStreams.fromPath(Paths.get(args[0]), StandardCharsets.UTF_8);

        MyLangLexer lexer = new MyLangLexer(input);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        MyLangParser parser = new MyLangParser(tokens);
        ParseTree tree = parser.program(); // parse

        EvalVisitor visitor = new EvalVisitor();
        visitor.visit(tree);

        printStates(visitor.getStates());
    }

    public static void printStates(Map<String, Integer> states) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");

        boolean first = true;
        for (Map.Entry<String, Integer> entry : states.entrySet()) {
            if (!first) {
                sb.append(", ");
            }
            sb.append("\"").append(entry.getKey()).append("\": ").append(entry.getValue());
            first = false;
        }
        sb.append("}");
        System.out.println(sb.toString());
    }
}