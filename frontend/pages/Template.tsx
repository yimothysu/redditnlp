import { Home as HomeIcon } from "lucide-react";

// Template component serves as a base for all pages of the website, providing a home button
export default function Template(props: any) {
  return (
    <div className="min-h-screen bg-gray-200">
      <div className="flex p-2">
        <a href="/" className="text-black!">
          <div className="flex ml-3 mt-2 gap-1 pt-1 pb-1 pl-2 pr-2 rounded-lg">
            <HomeIcon className="w-8 shadow-md h-8 rounded-full text-white bg-black p-1.5" />
            <h1 className="ml-1 mt-1">Home</h1>
          </div>
        </a>
      </div>
      {props.children}
    </div>
  );
}
