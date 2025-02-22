import { useParams } from "react-router-dom";
import Template from "./Template.tsx";

export default function RedditPage() {
  const { name } = useParams();

  return (
    <Template>
      <div className="text-center">
        <div className="p-4 font-bold">r/{name}</div>is coming soon!
      </div>
    </Template>
  );
}
