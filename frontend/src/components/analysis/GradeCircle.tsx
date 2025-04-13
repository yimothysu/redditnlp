interface GradeCircleProps {
  grade: string;
}

export const GradeCircle: React.FC<GradeCircleProps> = ({ grade }) => {
  const getGradeBackgroundColor = () => {
    if (["A+", "A", "A-"].includes(grade)) {
      return "bg-green-300";
    }
    if (["B+", "B", "B-"].includes(grade)) {
      return "bg-lime-300";
    }
    if (["C+", "C", "C-"].includes(grade)) {
      return "bg-yellow-300";
    }
    if (["D+", "D", "D-"].includes(grade)) {
      return "bg-orange-300";
    }
    if (["F"].includes(grade)) {
      return "bg-red-300";
    }
  };

  const getGradeBorderColor = () => {
    if (["A+", "A", "A-"].includes(grade)) {
      return "border-4 border-green-400";
    }
    if (["B+", "B", "B-"].includes(grade)) {
      return "border-4 border-lime-400";
    }
    if (["C+", "C", "C-"].includes(grade)) {
      return "border-4 border-yellow-400";
    }
    if (["D+", "D", "D-"].includes(grade)) {
      return "border-4 border-orange-400";
    }
    if (["F"].includes(grade)) {
      return "border-4 border-red-500";
    }
  };

  return (
    <span
      className={`inline-flex items-center justify-center w-7 h-7 ${getGradeBackgroundColor()} 
                ${getGradeBorderColor()} text-black text-[60%] m-3 rounded-full`}
    >
      {grade}
    </span>
  );
};
