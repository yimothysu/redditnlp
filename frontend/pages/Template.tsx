import { Home as HomeIcon } from "lucide-react";

export default function Template(props: any) {
  return (
    <div className="min-h-screen bg-gray-200">
      <div className="flex p-2">
        <a href="/" className="text-black!">
          <div className="flex ml-3 mt-2 ">
            {/* <b>Reddit</b>NLP */}
            <b className="mt-1">Home</b>
            <HomeIcon className="ml-1 w-7 h-7 text-gray-800" />
          </div>
        </a>
      </div>
      {props.children}
    </div>
  );
}
