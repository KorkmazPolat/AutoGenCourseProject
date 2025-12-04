
export const MOCK_TELEMETRY_DATA = {
    feedback: {
        total_events: 45,
        stages: {
            "research.generate": { approvals: 1, rejections: 0 },
            "course_planner.generate": { approvals: 1, rejections: 1 },
            "review_agent.course_plan": { approvals: 1, rejections: 1 },
            "lesson_writer.generate": { approvals: 5, rejections: 0 },
            "review_agent.lesson": { approvals: 4, rejections: 2 },
            "video_generation.queue": { approvals: 5, rejections: 0 },
            "video_generation.process": { approvals: 4, rejections: 1 }
        },
        failure_trends: {
            "course_planner.generate": 1,
            "review_agent.course_plan": 1,
            "review_agent.lesson": 2,
            "video_generation.process": 1
        },
        recent_failures: [
            {
                stage: "review_agent.lesson",
                metadata: { lesson: "Advanced Python", score: 4, feedback: "Too complex for beginners." },
                ts: Date.now() - 10000
            },
            {
                stage: "video_generation.process",
                metadata: { lesson: "Intro to AI", error: "Timeout waiting for renderer" },
                ts: Date.now() - 50000
            }
        ]
    },
    performance: {
        "research.generate": { count: 1, avg_seconds: 12.5, min_seconds: 12.5, max_seconds: 12.5 },
        "course_planner.generate": { count: 2, avg_seconds: 8.2, min_seconds: 7.5, max_seconds: 8.9 },
        "lesson_writer.generate": { count: 5, avg_seconds: 15.3, min_seconds: 10.1, max_seconds: 22.4 },
        "video_generator.generate": { count: 5, avg_seconds: 45.0, min_seconds: 30.0, max_seconds: 60.0 }
    },
    workload: {
        tasks_recorded: 5,
        successes: 4,
        failures: 1,
        max_queue_depth: 12,
        avg_queue_depth: 4.5,
        active_tasks: 3,
        task_log: [
            { task_name: "Lesson 1 Video", success: true, duration: 35.2 },
            { task_name: "Lesson 2 Video", success: true, duration: 42.1 },
            { task_name: "Lesson 3 Video", success: false, duration: 10.5, error: "Timeout" }
        ]
    }
};
