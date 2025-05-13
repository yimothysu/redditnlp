interface ColorCodeBoxProps {
  backgroundColor: string;
}

const ColorCodeBox: React.FC<ColorCodeBoxProps> = ({
  backgroundColor
}) => {
  return (
    <div
      className="flex items-center w-[70px] justify-center font-medium py-3 text-sm"
      style={{
        backgroundColor,
        textAlign: "center",
      }}
    >
    </div>
  );
};

// This component displays the color code key for sentiment 
export const ColorKey: React.FC = () => {
  if (window.innerWidth < 500) {
    return (
      <div className="mb-4">
        <p className="text-[12px] whitespace-pre-wrap text-center mb-1">
        <span className="text-red-600"><b>-1 (min) | negative consensus </b></span>                  <span className="text-green-600"><b>positive consensus | 1 (max)</b></span>
        </p>
        <div className="flex">
          <div className="bg-[#ff9898] h-5 w-20"></div>
          <div className="bg-[#ffc245] h-5 w-20"></div>
          <div className="bg-[#ffffac] h-5 w-20"></div>
          <div className="bg-[#d3ea84] h-5 w-20"></div>
          <div className="bg-[#91CC91] h-5 w-20"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="ml-33 flex flex-col align-items">
      <h1 className="text-[13px] text-gray-500 pb-1 pl-35">consensus scale (negative to positive)</h1>
      <div className="flex">
      <h1 className="pr-3 text-gray-500 font-semibold pb-1">-1 (min)</h1>
      <ColorCodeBox backgroundColor="#ff9898"/>
      <ColorCodeBox backgroundColor="#ffc245"/>
      <ColorCodeBox backgroundColor="#ffffac"/>
      <ColorCodeBox backgroundColor="#d3ea84"/>
      <ColorCodeBox backgroundColor="#91CC91"/>
      <h1 className="pl-3 text-gray-500 font-semibold pb-1">1 (max)</h1>
    </div>
    </div>

  );
};
