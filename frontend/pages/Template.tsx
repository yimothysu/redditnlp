import React from 'react';

export default function Template(Props: any) {
    return (
    <>
        <h1>This would be navigation bar -- or something consistent on each page</h1>
        {Props.children}
    </>
    );
}