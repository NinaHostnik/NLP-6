import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;

class translation {
    public static void main(String[] args) throws IOException, ParseException {
        //JSON parser object to parse read file
        JSONParser jsonParser = new JSONParser();
        FileReader reader = new FileReader("train.json");

        // writer to write text
        File fout = new File("translate22.txt");
        FileOutputStream fos = new FileOutputStream(fout);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fos));

        //Read JSON file
        Object data = jsonParser.parse(reader);
        JSONArray newData = (JSONArray) ((JSONObject) data).get("data");
        for (int i = 420; i < newData.size(); i++) {
            // we get a topic
            JSONObject current = (JSONObject) newData.get(i);

            // the title of the topic
            String title = (String) current.get("title");
            title = title.replace('_', ' ');
            writer.write("[title:]");
            writer.newLine();
            writer.write(title);
            writer.newLine();

            // paragraphs, questions, answers
            JSONArray pqa = (JSONArray) current.get("paragraphs");
            for (Object item : pqa) {
                // paragraph
                JSONObject paragraph = (JSONObject) item;
                writer.write("[context:]");
                writer.newLine();
                String context = (String) paragraph.get("context");
                writer.write(context);
                writer.newLine();

                // questions and answers
                JSONArray qa = (JSONArray) paragraph.get("qas");

                for (Object value : qa) {
                    JSONObject single = (JSONObject) value;
                    String question = (String) single.get("question");
                    writer.write("[question:]");
                    writer.newLine();
                    writer.write(question);
                    writer.newLine();
                    JSONArray answers = (JSONArray) single.get("answers");
                    Boolean isImpossible = (Boolean) single.get("is_impossible");

                    if (!isImpossible) {
                        for (Object a : answers) {
                            JSONObject an = (JSONObject) a;
                            String answer = (String) an.get("text");
                            writer.write("[answer:]");
                            writer.newLine();
                            writer.write(answer);
                            writer.newLine();
                        }
                    }
                }
            }
        }

    }
}
