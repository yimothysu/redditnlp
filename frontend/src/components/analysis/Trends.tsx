import { SubredditAnalysis } from "../../types/redditAnalysis";
import { EntityPicture } from "./EntityPicture";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    Label,
    CartesianGrid,
    ResponsiveContainer,
    TooltipProps
  } from "recharts";
  import { ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';

interface TrendsProps {
  analysis: SubredditAnalysis;
}
type date_to_sentiment = [string, number, string[]];
const COLORS: string[] = [
    "#1982C4", // sky blue
    "#F15BB5", // bright pink
    "#FF595E", // red-pink
    "#9B5DE5", // purple
    "#00C49A", // sea green
    "#FF924C", // tangerine
    "#00BBF9", // bright cyan
    "#D81159", // crimson
    "#FF007F", // neon pink
    "#FF6F61", // colar-pink
    "#FFCA3A", // golden yellow
    "#FDCB6E", // soft yellow-orange
    "#FFAB00",  // amber
  ];
  
const getColor = (index: number) => COLORS[index % COLORS.length];

const SentimentChart: React.FC<TrendsProps> = ({analysis}) => {
    if(analysis == null) { return; }
    
    type CustomTooltipProps = TooltipProps<ValueType, NameType> & {
        entity_name: string;
    };
    const CustomTooltip = ({
        active,
        payload,
        label,
        entity_name,
    }: CustomTooltipProps) => {
        if (!active || !payload || !payload.length) return null;
      
        return (
          <div className="bg-white p-3 opacity-100 rounded shadow-md text-sm z-50">
            <p className="font-semibold">Date: {label}</p>
            {payload.map((entry, idx) => {
              const fullData = mergedData[entity_name][label];
              if (!fullData) return null;
              const [sentiment, keyPoints] = fullData;
              return (
                <div key={idx} className="mt-2">
                  <p className="p-1"  style={{ color: entry.color }}>{sentiment.toFixed(3)}</p>
                  {keyPoints.length > 0 && <p className="p-1 text-gray-700">Key points:</p>}
                  <ul className="list-disc ml-4 text-gray-600">
                    {keyPoints.map((kp: string, i: number) => (
                      <li key={i}>{kp}</li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        );
      };

    const entity_to_sentiment_progression = GetTrends({analysis}) as Map<string, date_to_sentiment[]>;
    console.log('got entity_to_sentiment_progression')
    if(entity_to_sentiment_progression == null) {
        return;
    }

    type sent_and_key_pts = [number, string[]];
    const mergedData: Record<string, Record<string, sent_and_key_pts>> = {}; // key = entity name, value = (key = date, value = (sentiment, key_points)) 
    const dateToRecord: Record<string, Record<string, sent_and_key_pts>> = {}; // key = date, value = (key = entity name, value = (sentiment, key_points))
    for (const [entity, sentiment_progression] of entity_to_sentiment_progression.entries()) {
      for (const [date, sentiment, key_points] of sentiment_progression) {
        if (!dateToRecord[date]) {
          dateToRecord[date] = {};
        }
        dateToRecord[date][entity] = [sentiment, key_points]
      }
    }
    console.log('got dateToRecord: ', dateToRecord);
    const sortedDates = Object.keys(dateToRecord).sort(); 
    for (const date of sortedDates) {
        for (const [entity_name, sentiment_and_key_pts] of Object.entries(dateToRecord[date])) {
            if(!(entity_name in mergedData)) {
                mergedData[entity_name] = {}
            }
            mergedData[entity_name][date] = sentiment_and_key_pts;
        }
    }
    console.log('got mergedData: ', mergedData);
    const entityNames = Array.from(entity_to_sentiment_progression.keys());
    const chartData: Record<string, Record<string, number>> = {}; // key = entity name, value = (key = date, value = sentiment)
    for(const [entity_name, date_to_sent_and_key_pts] of Object.entries(mergedData)) {
        for(const [date, sent_and_key_pts] of Object.entries(date_to_sent_and_key_pts)) {
            if(!(entity_name in chartData)) {
                chartData[entity_name] = {}
            }
            chartData[entity_name][date] = sent_and_key_pts[0]; // getting just the sentiment 
        }
    } 
    console.log('got chartData: ', chartData);
    return (
        <div className="items-center justify-center space-between pb-100 pt-10" style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem' }}>
          {entityNames.map((entity_name, idx) => (
            <div className="rounded shadow-md bg-white w-full opacity-100 md:w-[calc(32%-0.5rem)]"
              key={entity_name}
              style={{
                padding: '1rem',
                borderRadius: '8px',
                // width: 'calc(30% - 0.5rem)', // two per row with 1rem gap
              }}
            >
              <div className="flex gap-6 items-center justify-center">
                <EntityPicture entity_name={entity_name}></EntityPicture>
                <h2 className="text-center font-semibold text-xl mt-4" style={{ color: getColor(idx), marginBottom: '1rem' }}>{entity_name}</h2>
              </div>
              <ResponsiveContainer width="100%" height={window.innerWidth < 640 ? 150 : 200} >
                <LineChart data={Object.entries(chartData[entity_name]).map(([date, value]) => ({date,value,}))} 
                           margin={{ top: 15, right: 5, bottom: 20, left: 5 }}>
                  <CartesianGrid stroke="#ccc" strokeDasharray="3 3" />
                  <XAxis dataKey="date" stroke="#333" tick={{ fontSize: 11, fill: '#888888'}}>
                    <Label value="Date" offset={7} fontSize={13} position="bottom" />
                  </XAxis>
                  <YAxis domain={[-1, 1]} stroke="#333" tick={{ fontSize: 11, fill: '#888888' }}>
                    <Label value="Sentiment Score" angle={-90} fontSize={13} dy={-40} dx={15} position="left" />
                  </YAxis>
                  <Tooltip position={{ y: 150, x: -50 }} 
                           wrapperStyle={{
                            backgroundColor: 'white',
                            border: '1px solid #ccc',
                            borderRadius: '8px',
                            padding: '10px',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                            opacity: 1,
                            pointerEvents: 'none',
                            width: '400px',
                            zIndex: 9999
                          }}
                           content={(props) => <CustomTooltip {...props} entity_name={entity_name}/>} />
                  <Line
                    connectNulls
                    type="linear"
                    dataKey="value"
                    stroke={getColor(idx)}
                    strokeWidth={3}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      );
};

function GetTrends({ analysis }: { analysis: SubredditAnalysis }) {
    if(analysis == null) { return; }
    console.log('inside GetTrends')
    /*
    entity_to_sentiment_progression = map where:
        - key = name of entity (type = string)
        - value = list of (date, sentiment, key_points) tuples 
    */
    let entity_to_sentiment_progression = new Map<string, date_to_sentiment[]>();
    const date_to_entities = analysis.top_named_entities // type = Record<string, NamedEntity[]>
    for(const [date, entities] of Object.entries(date_to_entities)) {
        for(const entity of entities) {
            const new_sentiment_progression = entity_to_sentiment_progression.get(entity.name) || [];
            entity_to_sentiment_progression.set(entity.name, [...new_sentiment_progression, [date, entity.sentiment, entity.key_points]]);
        }
    }
    console.log('entity_to_sentiment_progression: ', entity_to_sentiment_progression);
    // Only keep the key value pairs in entity_to_sentiment_progression that have >= 2 elements in value 
    const filtered_entity_to_sentiment_progression = new Map([...entity_to_sentiment_progression].filter(([_, sentiment_progression]) => sentiment_progression.length >= 4 ));
    console.log('filtered_entity_to_sentiment_progression: ', filtered_entity_to_sentiment_progression);
    return filtered_entity_to_sentiment_progression;
};

export const Trends: React.FC<TrendsProps> = ({ analysis }) => {
    return (
        <div>
            <div className="flex flex-col rounded shadow p-3 max-w-sm mx-auto">
                <h1 className="text-center text-gray-500 text-[14px] p-2">This page is for visualizing changes in sentiment for named entities. </h1>
                <h1 className="text-center text-gray-500 text-[14px] p-2">If you're interested in visiting posts that discuss these entities, visit the
                    'Named Entities' section.
                </h1>
            </div>
            <SentimentChart analysis={analysis} ></SentimentChart>
        </div>
    );
};