import React from "react";

export default function Template(props: any) {
  return (
    <div className="bg-gray-100 min-h-screen">
      <div className="flex p-2">
        <a href="/" className="text-black!">
          <b>Reddit</b>NLP
        </a>
      </div>
      {props.children}
    </div>
  );
}
