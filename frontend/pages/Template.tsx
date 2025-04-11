import { Home as HomeIcon } from "lucide-react";

// Template component serves as a base for all pages of the website, providing a home button
export default function Template(props: any) {
  return (
    <div className="min-h-screen bg-gray-200">
      <div className="flex p-2">
        <a href="/" className="text-black!">
          <div className="flex ml-3 mt-2 ">
          <HomeIcon className="w-7 h-7 text-gray-800" />
          <b className="ml-1 mt-1">Home</b>
          </div>
        </a>
      </div>
      {props.children}
    </div>
  );
}
